
import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F

def random_gt_points(masks,n_points,n_class):
    """
        masks: [h, w] one hot labels
    """
    masks_size = masks.shape
    # print('mask',masks.shape)
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
    return points_mask# (n_class,n_points,n_tokens)


    # [x1, y1, x2, y2] to binary mask
def bbox_to_mask(bbox: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(target_shape[1], target_shape[0])
    if bbox.sum() == 0:
        return mask
    mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    return mask


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        # print(mask.shape)
        
        bbox = torchvision.ops.masks_to_boxes(mask.long()).long().squeeze(0)
        bbox_mask = bbox_to_mask(bbox,target_shape=mask.shape[1:]).unsqueeze(0)
        bbox_mask = torch.cat([torch.ones_like(bbox_mask),bbox_mask],dim=0)
        point = random_gt_points(mask.squeeze(0),1,n_class=2)
        # print(bbox.shape,mask.shape,bbox_mask.shape,np.unique(mask),point)

        return {
            'inp': img,
            'gt': mask,
            'bbox': bbox,
            'bbox_mask':bbox_mask,
            'point':point
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)
    
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.inp_size
        outsize = self.inp_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # # final transform
        # img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # random filp
        if self.augment:
            img, mask = self._sync_transform(img,mask)
        else:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }