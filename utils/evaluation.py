import numpy as np
import torch
from math import pi, sqrt, exp
import os

from skimage import io, transform
from sklearn import metrics
from torchvision import transforms
import torch.nn.functional as F
from utils import binary
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter1d
from scipy.signal import convolve
import denseCRF
import denseCRF3D
from typing import Any, Dict, Generator, ItemsView, List, Tuple


from tqdm import tqdm
import argparse


# def densecrf3d(I, P, param):
#     """
#     input parameters:
#         I: a numpy array of shape [D, H, W, C], where C is the channel number
#            type of I should be np.uint8, and the values are in [0, 255]
#         P: a probability map of shape [D, H, W, L], where L is the number of classes
#            type of P should be np.float32
#         param: a tuple giving parameters of CRF. see the following two examples for details.
#     """
#     return denseCRF3D.densecrf3d(I, P, param)

def densecrf3d(image,mask):
    """
    input parameters:
        I: a numpy array of shape [D, H, W, C], where C is the channel number
           type of I should be np.uint8, and the values are in [0, 255]
        P: a probability map of shape [D, H, W, L], where L is the number of classes
           type of P should be np.float32
        param: a tuple giving parameters of CRF. see the following two examples for details.
    """
    # load initial labels, and convert it into an array 'prob' with shape [H, W, C]
    # where C is the number of labels
    # prob[h, w, c] means the probability of pixel at (h, w) belonging to class c.
    Iq = np.asarray(image)
    Iq_min, Iq_max = Iq.min(), Iq.max()
    Iq = (Iq - Iq_min) / (Iq_max-Iq_min) * 255.
    Iq = Iq.transpose((0,2,3,1)).astype(np.uint8)[:,:,:,0:1]
    prob = np.asarray(mask).transpose((0,2,3,1)).astype(np.float32)
    # print(Iq.shape,prob.shape)

    # probability map for each class

    dense_crf_param = {}
    dense_crf_param['MaxIterations'] = 2.0
    dense_crf_param['PosW'] = 0.5
    dense_crf_param['PosRStd'] = 2
    dense_crf_param['PosCStd'] = 2
    dense_crf_param['PosZStd'] = 2
    dense_crf_param['BilateralW'] = 1.0
    dense_crf_param['BilateralRStd'] = 2.0
    dense_crf_param['BilateralCStd'] = 2.0
    dense_crf_param['BilateralZStd'] = 2.0
    dense_crf_param['ModalityNum'] = 1
    dense_crf_param['BilateralModsStds'] = (5.0,)

    lab = denseCRF3D.densecrf3d(Iq, prob, dense_crf_param)
    return lab.astype(int)

def densecrf(image,mask):
    # load initial labels, and convert it into an array 'prob' with shape [H, W, C]
    # where C is the number of labels
    # prob[h, w, c] means the probability of pixel at (h, w) belonging to class c.
    Iq = np.asarray(image)
    Iq_min, Iq_max = Iq.min(), Iq.max()
    Iq = (Iq - Iq_min) / (Iq_max-Iq_min) * 255.
    Iq = Iq.transpose((1,2,0)).astype(np.uint8)
    prob = np.asarray(mask).transpose((1,2,0))


    w1    = 4.  # weight of bilateral term
    alpha = 50.    # spatial std
    beta  = 20    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 4.0   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    lab = denseCRF.densecrf(Iq, prob, param)
    return lab.astype(int)

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
            # label_slice = label[ind, :, :,:]

            #imshow(slice, "./out/" + case[:-4] + "_img_" + str(ind) + ".jpg", denormalize=False)
            #imshow(label_slice, "./out/" + case[:-4] + "_label_" + str(ind) + ".jpg", denormalize=False)

            x, y = slice.shape[1], slice.shape[2]
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
            box, box_mask = None, None
            if bbox is not None:
                box, box_mask = torch.from_numpy(bbox[ind]).unsqueeze(0).cuda(), torch.from_numpy(bbox_mask[ind]).unsqueeze(0).cuda()
            point = None
            if random_points is not None:
                point = torch.from_numpy(random_points[ind]).unsqueeze(0).cuda()
            # net.eval()
            with torch.no_grad():
                outputs, class_logits, final_attn_weight, _ = net(image=input,low_res=True,bbox_masks=box_mask,points=point)

                # pred = (output.softmax(1) * class_logits.unsqueeze(-1).sigmoid()).squeeze().cpu().numpy()
                cls_pred = class_logits.unsqueeze(-1).sigmoid()
                cls_pred[:,0] = 1.
                cls_prediction[ind] = cls_pred.squeeze().cpu().detach().numpy()
                # print(outputs.shape,gt_cls.shape)
                if use_gt:
                    out = torch.softmax(outputs, dim=1) *  gt_cls.to(outputs.device)#
                else:
                    threshould=0.5
                    out = torch.softmax(outputs, dim=1) #* (cls_pred > threshould) # gt_cls.to(outputs.device)#

                    # move the prob of the non-existing class to the background
                    # non_existing_points_prob = (out * (cls_pred < threshould)).sum(1)
                    # out[:,0:1,:,:]+=non_existing_points_prob

                    # out = out * (cls_pred >= threshould)

                # out = densecrf(input.squeeze().cpu().numpy(),out)

                #imshow(out,  "./out_1/" + case[:-4] + "_pre_" + str(ind) + ".jpg", denormalize=False)
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
                # print(pred.shape,prediction.shape)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        # net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    # prediction = densecrf3d(patch_size_image,prediction)
    # msak=[pred > 0.5]
    if use_gt: prediction = prediction.argmax(1) 
    else:
        threshould=0.5
        smooth_label = False
        # smooth the pred cls labels
        if smooth_label:
            kernel_size = 5
            # kernel = (np.ones(kernel_size) / kernel_size).reshape(kernel_size,1)
            kernel = np.array(gauss(kernel_size,1.7)).reshape(kernel_size,1)
            # print(kernel)
            cls_prediction = convolve(cls_prediction, kernel, mode='same')
            # print(cls_prediction)

        prediction = (prediction * (cls_prediction > threshould).reshape(*cls_prediction.shape,1,1)).argmax(1) 
    del image
    del patch_size_image
    metric_list = []
    for i in range(1, classes):
        # masks = prediction == i
        # mask_out = np.zeros_like(masks)
        mask_out = prediction == i
        # n_pics = mask_out.shape[-1]*mask_out.shape[-2]
        # for ind in range(prediction.shape[0]):
        #     mask = mask_out[ind]
        #     if mask.sum()>(0.3*n_pics):
        #         mask=0
        #     # mask, changed = remove_small_regions(mask, min_area, mode="holes")
        #     # unchanged = not changed
        #     # mask, changed = remove_small_regions(mask, min_area, mode="islands")
        #     # unchanged = unchanged and not changed
        #     mask_out[ind] = mask
        metric_list.append(calculate_metric_percase(mask_out, label == i))
    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    # cls metric
    # print(cls_label.shape,cls_prediction.shape)
    APs = metrics.average_precision_score(cls_label, cls_prediction[:,1:], average=None) 
    # print(APs,metric_list)
    return metric_list, APs
    

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    adopted form SAM
    """

    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True