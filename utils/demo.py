import gc
import torch
import numpy as np
import cv2
from torch.nn import functional as F
from os import listdir, makedirs, getcwd
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from ipywidgets import interact, widgets, FileUpload
from IPython.display import display
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from copy import deepcopy



def show_mask(mask, ax, random_color=False, alpha=0.95):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class PointPromptDemo:
    def __init__(self, model, dataroot):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.img_name = None
        self.gt = None
        
        ## load demo data
        self.dataroot = dataroot
        self.img_path = join(dataroot, 'imgs')
        self.gt_path = join(dataroot, 'gts_ts')
        self.gt_path_files = sorted(glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if isfile(join(self.img_path, basename(file)))]

    def show_mask(self, mask, ax, random_color=False, alpha=0.95):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([251/255, 252/255, 30/255, alpha])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @torch.no_grad()
    def infer(self, x, y):
        coords_1024 = np.array([[[
            x * 1024 / self.img_size[1],
            y * 1024 / self.img_size[0]
        ]]])
        coords_torch = torch.tensor(coords_1024, dtype=torch.float32).to(self.model.device)
        labels_torch = torch.tensor([[1]], dtype=torch.long).to(self.model.device)
        point_prompt = (coords_torch, labels_torch)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = point_prompt,
            auto_prompt = None,
            boxes = None,
            masks = None,
            use_auto_points = True
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_probs,
            size = self.img_size,
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

        seg = np.uint8(low_res_pred > 0.5)

        return seg

    def show(self, fig_size=3, alpha=0.95, scatter_size=25):

        assert self.image is not None, "Please set image first."
        fig, axes = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = False

        plt.tight_layout()

        for i in range(2):
            axes[i].imshow(self.image)
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Ground Truth')
            else:
                axes[i].set_title('Segmentation')

        def onclick(event):
            if event.inaxes == axes[1]:
                x, y = float(event.xdata), float(event.ydata)
                with torch.no_grad():
                    ## rescale x, y from canvas size to 1024 x 1024
                    seg = self.infer(x, y)

                for i in range(2):
                    axes[i].clear()
                    axes[i].imshow(self.image)
                    axes[i].axis('off')
                    axes[i].scatter(x, y, c='r', s=scatter_size)

                axes[1].set_title('Segmentation')
                self.show_mask(seg, axes[1], random_color=False, alpha=alpha)

                axes[0].set_title('Ground Truth')
                gt_label_id = self.get_label_id((x, y))
                if gt_label_id > 0:
                    gt_show = np.uint8(self.gt == gt_label_id)
                    self.show_mask(gt_show, axes[0], random_color=False, alpha=alpha)

                gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def set_image(self, image_index):
        image_path = join(self.img_path, basename(self.gt_path_files[image_index]))
        image = np.load(image_path)
        self.image = image
        self.img_size = image.shape[:2]
        self.img_name = basename(image_path)
        image_preprocess = self.preprocess_image(image)
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)
        
        gt_path = self.gt_path_files[image_index]
        gt = np.load(gt_path)
        gt_resize = cv2.resize(
            gt,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        self.gt = gt_resize

    def preprocess_image(self, image):
        img_resize = cv2.resize(
            image,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        )
        torch.Tensor(pixel_mean).view(-1, 1, 1)
        # Resizing
        img_resize = img_resize  / 255.
        pixel_mean = np.array([123.675/255., 116.28/255., 103.53/255.])
        pixel_std = np.array([58.395/255., 57.12/255., 57.375/255.])
        img_resize = (img_resize - pixel_mean) / pixel_std
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        return img_tensor
    
    def get_label_id(self, coords):
        x, y = coords
        label_id = self.gt[int(y), int(x)]

        return label_id

