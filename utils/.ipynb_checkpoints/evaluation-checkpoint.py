import numpy as np
import torch
from math import pi, sqrt, exp
import os

from skimage import io
from sklearn import metrics
import torch.nn.functional as F
from utils import binary
from typing import Any, Dict, Generator, ItemsView, List, Tuple


from tqdm import tqdm
import argparse


def calculate_metric_percase(pred, gt, is_test = False):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        if is_test:
            jaccard = binary.jc(pred, gt)
            asd = binary.assd(pred, gt)
        else:
            jaccard, asd = 0, 0
        return dice, hd95, jaccard, asd
    elif pred.sum() >= 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def process_coco(model, dataloader, num_classes,device, ignore_label=0):
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for batch in tqdm(dataloader):
        image, gt = batch['image'],batch['label']
        image, gt = image.to(device), gt.numpy()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(image)
        pred = np.array(output, dtype=int)
        gt[gt == ignore_label] = num_classes
        gt = gt - 1

        conf_matrix += np.bincount(
            (num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
            minlength=conf_matrix.size,
        ).reshape(conf_matrix.shape)

        acc = np.full(num_classes, np.nan, dtype=float)
        iou = np.full(num_classes, np.nan, dtype=float)
        tp = conf_matrix.diagonal()[:-1].astype(float)

        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(dataloader.dataset.classes):
            res[f"ACC-{name}"] = 100 * acc[i]
        print(res)


    acc = np.full(num_classes, np.nan, dtype=float)
    iou = np.full(num_classes, np.nan, dtype=float)
    tp = conf_matrix.diagonal()[:-1].astype(float)

    pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(float)
    class_weights = pos_gt / np.sum(pos_gt)
    pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(float)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    union = pos_gt + pos_pred - tp
    iou_valid = np.logical_and(acc_valid, union > 0)
    iou[iou_valid] = tp[iou_valid] / union[iou_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)

    res = {}
    res["mIoU"] = 100 * miou
    res["fwIoU"] = 100 * fiou
    res["mACC"] = 100 * macc
    res["pACC"] = 100 * pacc
    for i, name in enumerate(dataloader.dataset.classes):
        res[f"ACC-{name}"] = 100 * acc[i]
    print(res)




@torch.no_grad()
def process_med(model, dataloader, num_classes, device, min_area=100):
    label_list = []
    num_samples = len(dataloader.dataset)
    dice = np.zeros(num_classes-1)
    hd95 = np.zeros(num_classes-1)
    jaccard = np.zeros(num_classes-1)
    asd = np.zeros(num_classes-1)
    for batch in tqdm(dataloader):
        image, gt = batch['image'],batch['label']
        image, gt = image.to(device), gt.numpy()
        label_list.append(batch['multi_hot'])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # ori_res_masks, class_features, final_attn_weight_list, feature_with_pe_list
            output, class_logits, _, _ = model(image)
        
        cls_pred = class_logits.unsqueeze(-1).sigmoid()
        cls_pred[:,0] = 1.
        pred = (output.softmax(1) * cls_pred).squeeze().cpu().numpy()
        for i in range(gt.shape[0]):
            cur_pred = pred[i]
            cur_img = image[i]
            # cur_pred = densecrf(cur_img.cpu().numpy(),cur_pred)
            cur_pred = pred[i].argmax(0)
            for j in range(1,num_classes):
                # print((gt[i]==j).shape,(pred[i]==j).shape)mask, changed = remove_small_regions(mask, min_area, mode="holes")
                
                
                mask = cur_pred==j
                
                # mask, changed = remove_small_regions(mask, min_area, mode="holes")
                # # unchanged = not changed
                # mask, changed = remove_small_regions(mask, min_area, mode="islands")
                # unchanged = unchanged and not changed
                cur_di,cur_hd,cur_ja,cur_as = calculate_metric_percase(mask,gt[i]==j)
                # print(cur_di,cur_hd,cur_ja,cur_as)
                dice[j-1]+=cur_di
                hd95[j-1]+=cur_hd
                jaccard[j-1]+=cur_ja
                asd[j-1]+=cur_as
    
    dice /= num_samples
    hd95 /= num_samples
    jaccard /= num_samples
    asd /= num_samples

    print(f'Dice: {dice}, hd95: {hd95}, jaccard: {jaccard}, asd: {asd}')

    return dice.mean()



@torch.no_grad()
def test_med(model, dataloader, num_classes, device, min_area=100,use_gt=False,use_bbox=False,use_random_points=False,test_save_path=None):
    # dice = np.zeros(num_classes-1)
    # hd95 = np.zeros(num_classes-1)
    # jaccard = np.zeros(num_classes-1)
    # asd = np.zeros(num_classes-1)
    metric_list = 0.0
    AP_list = 0.0
    size = [256,256]
    for i,batch in enumerate(tqdm(dataloader)):
        image, gt = batch['image'],batch['label'].float()
        multi_hot = batch['multi_hot']
        # volume_path = batch['path']
        bbox = None
        bbox_mask = None
        if use_bbox:
            bbox = batch['bbox']
            bbox_mask = batch['bbox_mask']
        random_points = None
        if use_random_points:
            random_points = batch['point']
        gt = F.interpolate(
            gt.permute(1,0,2,3),
            size=size,
            mode="nearest-exact",
        ).permute(1,0,2,3).long()
        # print(gt.shape)
        # print(gt.unique())
        # gt = F.one_hot(gt,num_classes=num_classes).permute(0,1,4,2,3)
        # print(gt.unique())
        metric_i, APs_i = test_single_volume(image=image, label=gt, cls_label=multi_hot, net=model, classes=num_classes, patch_size=size, test_save_path=test_save_path, case=batch['path'][0], z_spacing=1,use_gt=use_gt,bbox=bbox,bbox_mask=bbox_mask, random_points=random_points)
        metric_list += np.array(metric_i)
        AP_list += np.array(APs_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f mAP %f' % (i, batch['path'], np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],np.mean(APs_i, axis=0)))
    
    metric_list = metric_list / len(dataloader)
    AP_list = AP_list / len(dataloader)
    for i in range(1, num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f AP %f' % (i, metric_list[i-1][0], metric_list[i-1][1], AP_list[i-1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mAP = np.mean(AP_list)
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mAP: %f' % (performance, mean_hd95, mAP))
    print("Testing Finished!")
    return performance, mean_hd95

def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

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

def test_single_volume(image, label, cls_label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, min_area=100,use_gt=False, bbox = None, bbox_mask=None,random_points=None):
    # print(image.shape,label.shape)
    patch_size_image = F.interpolate(
                image.squeeze(0),
                size=patch_size,
                mode="bilinear",
            ).cpu().detach().numpy()
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    torch_cls = cls_label
    # print(torch_cls.shape)
    cls_label = cls_label.squeeze(0).cpu().detach().numpy()
    n,h,w = label.shape
    if bbox is not None:
        bbox, bbox_mask = bbox.squeeze(0).cpu().detach().numpy(), bbox_mask.squeeze(0).cpu().detach().numpy()
    if random_points is not None:
        random_points = random_points.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 4:
        prediction = np.zeros((n,classes,h,w))
        cls_prediction = np.zeros((n,classes))
        for ind in range(image.shape[0]):
            slice = image[ind, :, :,:]
            gt_cls = torch_cls[:,ind].unsqueeze(-1).unsqueeze(-1)
            x, y = slice.shape[1], slice.shape[2]
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
            box, box_mask = None, None
            if bbox is not None:
                box, box_mask = torch.from_numpy(bbox[ind]).unsqueeze(0).cuda(), torch.from_numpy(bbox_mask[ind]).unsqueeze(0).cuda()
            point = None
            if random_points is not None:
                point = torch.from_numpy(random_points[ind]).unsqueeze(0).cuda()
            with torch.no_grad():
                outputs, class_logits, final_attn_weight, _ = net(image=input,low_res=True,bbox_masks=box_mask,points=point)

                cls_pred = class_logits.unsqueeze(-1).sigmoid()
                cls_pred[:,0] = 1.
                cls_prediction[ind] = cls_pred.squeeze().cpu().detach().numpy()
                if use_gt:
                    out = torch.softmax(outputs, dim=1) *  gt_cls.to(outputs.device)
                else:
                    threshould=0.5
                    out = torch.softmax(outputs, dim=1) 

                if case is not None and test_save_path is not None:
                    onehot_points = general_to_onehot(final_attn_weight).cpu().numpy()
                    prediction_mask = out.cpu().numpy()
                    slice_path = os.path.join(test_save_path,f"{case.split('.')[0]}_{ind}.npz")
                    np.savez(slice_path,points=onehot_points,pred=prediction_mask)

                if x != patch_size[0] or y != patch_size[1]:
                    pred = F.interpolate(
                        out,
                        size=patch_size,
                        mode="bilinear",
                    ).squeeze(0).cpu().detach().numpy()
                else:
                    pred = out.squeeze(0).cpu().detach().numpy()
                prediction[ind] = pred
    else:
        raise NotImplementedError
    
    if use_gt: prediction = prediction.argmax(1) 
    else:
        threshould=0.5

        prediction = (prediction * (cls_prediction > threshould).reshape(*cls_prediction.shape,1,1)).argmax(1) 
    del image
    del patch_size_image
    metric_list = []
    for i in range(1, classes):
        mask_out = prediction == i
        metric_list.append(calculate_metric_percase(mask_out, label == i))
    
    APs = metrics.average_precision_score(cls_label, cls_prediction[:,1:], average=None) 
    return metric_list, APs
    