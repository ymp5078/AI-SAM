# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join

import time
from tqdm import tqdm
from skimage import transform
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.processed_ACDC import generate_dataset as generate_acdc_dataset
from datasets.lidc import generate_dataset as generate_lidc_dataset
from datasets.synapse import generate_dataset as generate_synapse_dataset
from utils.evaluation import process_med, test_med
from ai_sam.ai_sam import AISAM
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()


os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_path",
    type=str,
    default="/data/yimu/coco",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("--dataset", type=str, default="acdc")
parser.add_argument("--img_size", type=int, default=1024)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("-task_name", type=str, default="ai-sam-ViT-H")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="/home/ymp5078/Segmentations/SamAdapter/weights/mobile_sam.pt"
)


parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-num_workers", type=int, default=5)
# Optimizer parameters
parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--device", type=str, default="cuda:0")

# model settings
parser.add_argument("--use_hard_points", action="store_true", default=False, help='if use hard threshould for points')
parser.add_argument("--use_classification_head", action="store_true", default=False, help='if also perform classification')
parser.add_argument("--use_lora", action="store_true", default=False, help='if use lora for the encoder')
parser.add_argument("--use_gt", action="store_true", default=False, help='if use gt for cls label')
parser.add_argument("--normalize", action="store_true", default=False, help='if use SAM norm')
parser.add_argument("--return_bbox", action="store_true", default=False, help='if use SAM norm')
parser.add_argument("--n_random_points", type=int, default=0, help='num of random gt points to use, 0 mean use auto points')
parser.add_argument("--save_path", type=str, default=None, help='where to save the output files')

args = parser.parse_args()

# %% set up model for training
device = torch.device(args.device)

def main():
    
    # dataset
    if args.dataset=='acdc':
        train_dataloader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_acdc_dataset(args)
        num_classes = train_dataloader.dataset.NUM_CLASS
    elif args.dataset=='synapse':
        train_dataloader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_synapse_dataset(args)
        num_classes = train_dataloader.dataset.NUM_CLASS
    print('NUM_CLASS:',num_classes)
    sam_checkpoint = '/scratch/bbmr/ymp5078/segmentations/weights/sam_vit_b_01ec64.pth' if args.model_type == 'vit_b' else '/scratch/bbmr/ymp5078/segmentations/weights/sam_vit_h_4b8939.pth'
    ai_sam_model = AISAM(
        num_classes=num_classes,
        sam_checkpoint=sam_checkpoint,
        sam_model=args.model_type,
        num_class_tokens=16,
        num_points=4,
        use_classification_head=args.use_classification_head,
        use_hard_points=args.use_hard_points,
        use_lora = args.use_lora
    ).to(device)
    ## Map model to be loaded to specified single GPU
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ai_sam_model.load_state_dict(checkpoint["model"],strict=True)
    ai_sam_model.eval()
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    if args.use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            test_med(model=ai_sam_model,dataloader=test_loader,num_classes=num_classes,test_save_path=args.save_path,device=device,use_gt=args.use_gt,use_bbox=args.return_bbox,use_random_points=args.n_random_points > 0)
    else:
        test_med(model=ai_sam_model,dataloader=test_loader,num_classes=num_classes,test_save_path=args.save_path,device=device,use_gt=args.use_gt,use_bbox=args.return_bbox,use_random_points=args.n_random_points > 0)


    



if __name__ == "__main__":
    main()
