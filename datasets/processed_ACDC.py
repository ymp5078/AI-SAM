#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure
from skimage.util import random_noise
from torch.utils.data import Dataset
import torchvision
join = os.path.join


def random_gt_points(masks,n_points,n_class):
    """
        masks: [h, w] one hot labels
    """
    masks_size = masks.shape
    masks_flatten = masks.flatten() # [h*w]
    n_tokens = masks_flatten.shape[0]
    points_mask = np.zeros((n_class,n_points,2),dtype=int)
    for i in range(n_class):
        candidates = np.arange(n_tokens)[masks_flatten==i]
        if len(candidates) > 0:
            points = np.random.choice(candidates,n_points,replace=False)
        else:
            points = np.zeros(n_points)
        points_x = points % masks_size[1]
        points_y = points // masks_size[0]
        points_mask[i,:,0] = points_x
        points_mask[i,:,1] = points_y
    # print(masks.shape)
    # print(points_mask[1,:,0],points_mask[1,:,1])
    # print(masks[points_mask[1,:,0],points_mask[1,:,1]])
    # print(masks[points_mask[2,:,0],points_mask[2,:,1]])
    return points_mask# (n_class,n_points,n_tokens)


    # [x1, y1, x2, y2] to binary mask
def bbox_to_mask(bbox: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(target_shape[1], target_shape[0])
    if bbox.sum() == 0:
        return mask
    mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    return mask

def random_crop(image,label):
    x, y = image.shape
    x_crop_size = int(np.random.uniform(0.7,1.0) * x)
    x_crop_start = np.random.randint(0,x-x_crop_size)
    y_crop_size = int(np.random.uniform(0.7,1.0) * y)
    y_crop_start = np.random.randint(0,y-y_crop_size)
    
    image = image[x_crop_start:x_crop_start+x_crop_size,y_crop_start:y_crop_start+y_crop_size]
    label = label[x_crop_start:x_crop_start+x_crop_size,y_crop_start:y_crop_start+y_crop_size]
    return image, label
    

    

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def normalize(img,pixel_mean,pixel_std):
    # # normalize to SAM norm
    img = (img - pixel_mean) / pixel_std
    return img


class RandomGenerator(object):
    def __init__(self, output_size,normalize=True):
        self.normalize = normalize
        self.output_size = output_size
        pixel_mean = [123.675/255., 116.28/255., 103.53/255.],
        pixel_std = [58.395/255., 57.12/255., 57.375/255.],
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_crop(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).expand((3,-1,-1))
        if self.normalize:
            image = normalize(image,self.pixel_mean,self.pixel_std)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
    
class Resize(object):
    NUM_CLASS = 4
    def __init__(self, output_size, multi_slices=False,normalize=True, return_bbox=False,n_points=0):
        self.normalize = normalize
        self.output_size = output_size
        self.return_bbox = return_bbox
        self.n_points = n_points
        pixel_mean = [123.675/255., 116.28/255., 103.53/255.],
        pixel_std = [58.395/255., 57.12/255., 57.375/255.],
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.multi_slices=multi_slices

    def __call__(self, sample):
        if self.multi_slices:
            images, labels = sample['image'], sample['label']
            image_list, label_list = [], []
            point_list = []
            if self.return_bbox:
                bbox_list, bbox_mask_list = [], []
            for i in range(images.shape[0]):
                image, label = images[i],labels[i]
                x, y = image.shape
                if x != self.output_size[0] or y != self.output_size[1]:
                    image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)  # why not 3?
                    label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
                image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).expand((3,-1,-1))
                if self.normalize:
                    image = normalize(image,self.pixel_mean,self.pixel_std)
                if self.n_points > 0:
                    points = random_gt_points(masks=label.astype(int),n_points=self.n_points,n_class=self.NUM_CLASS)
                    point_list.append(torch.from_numpy(points.astype(np.float32)))
                label = torch.from_numpy(label.astype(np.float32))
                image_list.append(image)
                label_list.append(label)
                if self.return_bbox:
                    bbox_per_class,bbox_mask_per_class = [], []
                    for i in range(self.NUM_CLASS):
                        cls_mask = (label.long()==i).unsqueeze(0).long()
                        if cls_mask.sum()==0:
                            bbox = torch.tensor([0,0,0,0])
                        else:
                            bbox = torchvision.ops.masks_to_boxes(cls_mask).long().squeeze(0)
                        bbox_mask = bbox_to_mask(bbox,target_shape=label.shape)
                        bbox_per_class.append(bbox)
                        bbox_mask_per_class.append(bbox_mask)
                    bbox_list.append(torch.stack(bbox_per_class,dim=0))
                    bbox_mask_list.append(torch.stack(bbox_mask_per_class,dim=0))
                
            images = torch.stack(image_list,dim=0)
            labels = torch.stack(label_list,dim=0)
            sample = {'image': images, 'label': labels.long()}
            if len(point_list) > 0:
                points = torch.stack(point_list,dim=0)
                sample['point'] = points
            if self.return_bbox:
                bboxs = torch.stack(bbox_list,dim=0)
                bbox_masks = torch.stack(bbox_mask_list,dim=0)
                sample['bbox'] = bboxs
                sample['bbox_mask'] = bbox_masks
        else:
            image, label = sample['image'], sample['label']
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)  # why not 3?
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).expand((3,-1,-1))
            if self.normalize:
                image = normalize(image,self.pixel_mean,self.pixel_std)
            if self.n_points > 0:
                points = random_gt_points(masks=label.astype(int),n_points=self.n_points,n_class=self.NUM_CLASS)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
            if self.n_points > 0:
                sample['point'] = torch.from_numpy(points)
            if self.return_bbox:
                bbox_per_class,bbox_mask_per_class = [], []
                for i in range(self.NUM_CLASS):
                    cls_mask = (label.long()==i).unsqueeze(0).long()
                    if cls_mask.sum()==0:
                        bbox = torch.tensor([0,0,0,0])
                    else:
                        bbox = torchvision.ops.masks_to_boxes(cls_mask).long().squeeze(0)
                    bbox_mask = bbox_to_mask(bbox,target_shape=label.shape)
                    bbox_per_class.append(bbox)
                    bbox_mask_per_class.append(bbox_mask)
                sample['bbox'] = torch.stack(bbox_per_class,dim=0)
                sample['bbox_mask'] = torch.stack(bbox_mask_per_class,dim=0)
            
        return sample


class ACDCdataset(Dataset):
    NUM_CLASS = 4
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split

        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, self.split, slice_name)
        data = np.load(data_path)
        image, label = data['img'], data['label']
        # from [-1, 1] to [0,1] 
        image = (image + 1.)/2
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['path'] = self.sample_list[idx].strip('\n').replace('.npz','.jpg')
        if self.split == "train" or self.split == 'valid':
            unique_classes = np.unique(sample['label'])
            # not remove background
            multi_hot = self._label_list_to_multi_hot(unique_classes,self.NUM_CLASS)#[1:self.NUM_CLASS]
            # general resize, normalize and toTensor
            class_num = np.random.choice(self.NUM_CLASS-1,1)[0] # remove background
            class_num = class_num + 1
            one_class_mask = torch.eq(sample['label'],class_num).long().unsqueeze(0)
            sample['original_size'] = sample['image'].shape[1:]
            sample['multi_hot'] = multi_hot
            sample['sampled_class'] = class_num
            sample['sampled_mask'] = one_class_mask
        else:
            n_slices = sample['label'].shape[0]
            multi_hots = np.zeros((n_slices,self.NUM_CLASS))
            for i in range(n_slices):
                unique_classes = np.unique(sample['label'][i])
                # not remove background
                multi_hot = self._label_list_to_multi_hot(unique_classes,self.NUM_CLASS)#[1:self.NUM_CLASS]
                multi_hots[i]=multi_hot
                # general resize, normalize and toTensor
            class_num = np.random.choice(self.NUM_CLASS-1,1)[0] # remove background
            class_num = class_num + 1
            one_class_mask = torch.eq(sample['label'][i],class_num).long().unsqueeze(0)
            sample['original_size'] = sample['image'].shape[1:]
            sample['multi_hot'] = multi_hots
            sample['sampled_class'] = class_num
            sample['sampled_mask'] = one_class_mask
            
        return sample
    
    def _label_list_to_multi_hot(self, labels, num_classes):
        """Converts a label list to a multi-hot tensor.

        Args:
            labels: A list of labels.
            num_classes: The number of classes.

        Returns:
            A multi-hot tensor of shape (num_classes).
        """

        multi_hot = np.zeros(num_classes)
        multi_hot[labels] = 1

        return multi_hot


def generate_dataset(args):

    train_transform = RandomGenerator(output_size=[args.img_size, args.img_size],normalize=args.normalize)
    val_transform = Resize(output_size=[args.img_size, args.img_size],normalize=args.normalize,return_bbox=args.return_bbox,n_points=args.n_random_points)
    test_transform = Resize(output_size=[args.img_size, args.img_size],multi_slices=True,normalize=args.normalize,return_bbox=args.return_bbox,n_points=args.n_random_points)
    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        # args.img_size = 224
        train_ds = ACDCdataset(base_dir=args.tr_path, list_dir=join(args.tr_path,'lists_ACDC'), split='train', transform=train_transform)
        val_ds = ACDCdataset(base_dir=args.tr_path, list_dir=join(args.tr_path,'lists_ACDC'), split='valid', transform=val_transform)
        test_ds = ACDCdataset(base_dir=args.tr_path, list_dir=join(args.tr_path,'lists_ACDC'), split='test', transform=test_transform)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    train_sampler = None
    val_sampler = None
    test_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler