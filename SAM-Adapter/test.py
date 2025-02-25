import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

from visualize import get_color_pallete, vocpallete

join = os.path.join

def visualize(save_path,image,pre_masks,gt_seg,pre_points,bbox,ind):
    # print(pre_points)
    fig,ax = plt.subplots(figsize=(5, 5))
    # print(image.size,pre_masks.shape,gt_seg.shape,pre_points.shape)
    ax.imshow(image)
    if bbox is not None:
        # print(bbox.shape)
        x,y,x2,y2 = bbox.numpy()
        w = x2 - x
        h = y2 - y
        # print([[v/255. for v in vocpallete[c_id*3:c_id*3+3]]]*num_points)
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=[0.9,0,0], facecolor='none')
        ax.add_patch(rect)
    if pre_points is not None:
        num_classes = pre_points.shape[1]
        for c_id in range(num_classes):
            cur_points = general_to_onehot(pre_points[0,c_id]).cpu().numpy() * np.array(image.size)
            # print(cur_points)
            num_points = len(cur_points)
            plt.scatter(cur_points[:,0],cur_points[:,1],s=20,c=[[v/255. for v in vocpallete[c_id*3:c_id*3+3]]]*num_points,linewidths=0.5,edgecolors=[0.99,0.99,0.99])
        
    plt.axis('off')
    plt.savefig(join(save_path, f"{ind}.png"),bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(5, 5))
    gt_seg = get_color_pallete(gt_seg.squeeze()).convert('RGB')
    plt.imshow(gt_seg)
    plt.axis('off')
    plt.savefig(join(save_path, f"{ind}_gt.png"),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5, 5))
    pre_seg = get_color_pallete((pre_masks > 0.5).squeeze()).convert('RGB')
    plt.imshow(pre_seg)
    plt.axis('off')
    plt.savefig(join(save_path, f"{ind}_pred.png"),bbox_inches='tight')
    plt.close()


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

def general_to_onehot(g_points):
    """
        assume the last two dimensions are spatial, h and w
        return the relative position of the point [0,1]
    """
    g_point_size = g_points.shape[-2:]
    g_points = g_points.flatten(-2).argmax(-1)
    points_x = (g_points % g_point_size[1]) / g_point_size[1]
    points_y = (g_points // g_point_size[0]) / g_point_size[0]
    points = torch.stack([points_x,points_y],dim=-1)
    return points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, return_points=False,test_save_path=None):
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
    elif eval_type == 'dice':
        metric_fn = utils.calc_dice
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for ind, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        if not return_points:
            pred = torch.sigmoid(model.infer(inp))
            final_attn_weight = None
        else:
            pred,final_attn_weight = model.infer(inp,return_points=return_points)
            pred = torch.sigmoid(pred)
        if test_save_path is not None:

            mean=torch.tensor([0.485, 0.456, 0.406])[:,None,None]
            std=torch.tensor([0.229, 0.224, 0.225])[:,None,None]
            pil_img = transforms.functional.to_pil_image(inp.cpu()[0]*std+mean)
            visualize(save_path=test_save_path,image=pil_img,pre_masks=pred.cpu().numpy()[0],pre_points=final_attn_weight,gt_seg=batch['gt'].cpu().numpy()[0],bbox=None,ind=ind)

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

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--save_path', default=None)
    parser.add_argument("--return_points", action="store_true", default=False, help='if plot autoprompt')
    # parser.add_argument('--return_points', default=None)
    args = parser.parse_args()

    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    with torch.no_grad():
        metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                        data_norm=config.get('data_norm'),
                                                        eval_type=config.get('eval_type'),
                                                        eval_bsize=config.get('eval_bsize'),
                                                        verbose=True,
                                                        return_points=args.return_points,
                                                        test_save_path=args.save_path)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
