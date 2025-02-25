import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


# def eval_psnr(loader, model, eval_type=None):
#     model.eval()

#     if eval_type == 'f1':
#         metric_fn = utils.calc_f1
#         metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
#     elif eval_type == 'fmeasure':
#         metric_fn = utils.calc_fmeasure
#         metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
#     elif eval_type == 'ber':
#         metric_fn = utils.calc_ber
#         metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
#     elif eval_type == 'cod':
#         metric_fn = utils.calc_cod
#         metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

#     if local_rank == 0:
#         pbar = tqdm(total=len(loader), leave=False, desc='val')
#     else:
#         pbar = None

#     pred_list = []
#     gt_list = []
#     for batch in loader:
#         for k, v in batch.items():
#             batch[k] = v.cuda()

#         inp = batch['inp']
        
#         pred = torch.sigmoid(model.infer(inp)).detach()
#         pred_list.append(pred.cpu())
#         gt_list.append(batch['gt'].cpu())

#         # FIXME: OOM
#         # batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
#         # batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

#         # dist.all_gather(batch_pred, pred)
#         # pred_list.extend(batch_pred)
#         # dist.all_gather(batch_gt, batch['gt'])
#         # gt_list.extend(batch_gt)
#         if pbar is not None:
#             pbar.update(1)

#     if pbar is not None:
#         pbar.close()

#     pred_list = torch.cat(pred_list, 0)
#     gt_list = torch.cat(gt_list, 0)
#     result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

#     return result1, result2, result3, result4, metric1, metric2, metric3, metric4

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])
        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
        
    results = torch.tensor([val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()]).cuda()
    batch_results = [torch.zeros_like(results) for _ in range(dist.get_world_size())]

    dist.all_gather(batch_results, results)
    results = torch.stack(batch_results,dim=0).mean(0).cpu().numpy()
    # print(*results)
        # print(val_metric1.n)

    return *results, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        auto_prompt_params = list(model.auto_prompt.parameters())

        mask_decoder_params = list(model.mask_decoder.parameters())

        trainable_params = auto_prompt_params+mask_decoder_params
        optimizer = utils.make_optimizer(
            trainable_params, config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        auto_prompt_params = list(model.auto_prompt.parameters())

        mask_decoder_params = list(model.mask_decoder.parameters())

        trainable_params = auto_prompt_params+mask_decoder_params
        optimizer = utils.make_optimizer(
            trainable_params, config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    verbose = True
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.verbose = verbose
        model.set_input(inp, gt)
        model.optimize_parameters()
        verbose = False
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            log_dict = {
                'loss': model.loss_G.item(),
                'attn_loss':model.loss_attn.item(),
                'lr':model.optimizer.param_groups[0]['lr']}
            pbar.set_postfix(log_dict)
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    print('before:',model.pe_layer.positional_encoding_gaussian_matrix.mean())
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    print('after:',model.pe_layer.positional_encoding_gaussian_matrix.mean())
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "pe_layer" in name:
            para.requires_grad_(False)
    for para in model.pe_layer.parameters():
        print(para.requires_grad)
    # print(model.prompt_encoder.requires_grad)
            
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local-rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
