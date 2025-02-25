import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

from models.segment_anything import sam_model_registry
from visualize import get_color_pallete, vocpallete

join = os.path.join

def visualize(save_path,image,bbox,ind):
    # print(pre_points)
    fig,ax = plt.subplots(figsize=(5, 5))
    # print(bbox.shape)
    # print(image.size,pre_masks.shape,gt_seg.shape,pre_points.shape)
    ax.imshow(image)
    if bbox is not None:
        # print(bbox.shape)
        x,y,x2,y2 = bbox[0]
        w = x2 - x
        h = y2 - y
        # print([[v/255. for v in vocpallete[c_id*3:c_id*3+3]]]*num_points)
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=[v/255. for v in vocpallete[1*3:1*3+3]], facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(join(save_path, f"{ind}_box.png"),bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(join(save_path, f"{ind}_image.png"),bbox_inches='tight')
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, use_bbox=False, n_random_points=0,
              verbose=False,base_sam=False, return_points=False,test_save_path=None):
    
    pbar = tqdm(loader, leave=False, desc='val')

    for ind, batch in enumerate(pbar):

        inp = batch['inp']
        bbox = batch['bbox']
        # print(bbox.shape)
        bbox_mask = batch['bbox_mask'] if use_bbox else None
        points = batch['point'] if n_random_points > 0 else None

        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        if test_save_path is not None:
            mean=torch.tensor([0.485, 0.456, 0.406])[:,None,None]
            std=torch.tensor([0.229, 0.224, 0.225])[:,None,None]
            pil_img = transforms.functional.to_pil_image(inp.cpu()[0]*std+mean)
            visualize(save_path=test_save_path,image=pil_img,bbox=bbox.cpu().numpy(),ind=ind)



    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument("--use_bbox", action="store_true", default=False, help='if use bbox')
    parser.add_argument("--n_random_points", type=int, default=0, help='num of random gt points to use, 0 mean use auto points')
    parser.add_argument("--use_base_sam", action="store_true", default=False, help='if use base SAM model')
    parser.add_argument('--save_path', default=None)
    parser.add_argument("--return_points", action="store_true", default=False, help='if plot autoprompt')
    
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
        
    eval_psnr(loader, None,
            data_norm=config.get('data_norm'),
            eval_type=config.get('eval_type'),
            eval_bsize=config.get('eval_bsize'),
            use_bbox=args.use_bbox,
            n_random_points=args.n_random_points,
            verbose=True,base_sam=args.use_base_sam,
            return_points=args.return_points,
            test_save_path=args.save_path)
