import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer, PromptEncoder
# from .segment_anything import sam_model_registry
from .transformer_dev_2 import ModifiedTwoWayTransformer, MLP, ModifiedAttention, ModifiedTransformer, LayerNorm2d
from .attn_losses import AttnMaskLoss

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('auto_prompt_sam')
class AutoPromptSAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.verbose = False
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']

        self.auto_prompt = AutoPrompt(
            num_classes=2,
            transformer_dim=self.prompt_embed_dim,
            embed_dim = self.prompt_embed_dim,
            num_class_tokens=encoder_mode['num_class_tokens'],
            num_points=encoder_mode['num_points'],
            use_auto_points=True,
            use_classification_head=False,
            use_feature_prompt=False,
            num_adapt_emb=0,
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=(inp_size//encoder_mode['patch_size'], inp_size//encoder_mode['patch_size']),
            input_image_size=(inp_size, inp_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False



        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        # self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.pe_layer = self.prompt_encoder.pe_layer
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])


        self.criterionAttn = AttnMaskLoss(gamma=0.1,temp=7, include_background=True)

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
                    
        return self.pe_layer((self.image_embedding_size,self.image_embedding_size)).unsqueeze(0)#.detach()
        # return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs = self.input.shape[0]
       # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        points_type = torch.cat([
            self.prompt_encoder.not_a_point_embed.weight,
            self.prompt_encoder.point_embeddings[0].weight,
            self.prompt_encoder.point_embeddings[1].weight,
            ]).detach()

        self.features = self.image_encoder(self.input)

        auto_prompt_list, class_features, final_attn_weight_list, feature_list, feature_with_pe_list = self.auto_prompt(
            image=self.features.detach(), 
            image_pe=self.get_dense_pe().detach(), 
            points_type=points_type,
            hard=False,
        )
        sparse_embeddings = torch.cat((sparse_embeddings, auto_prompt_list[1]), dim=1)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks
        self.class_features, self.final_attn_weight_list, self.feature_with_pe_list = class_features, final_attn_weight_list, feature_with_pe_list

    def infer(self, input,bbox_mask=None,points=None,return_points=False):
        bs = input.shape[0]

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        points_type = torch.cat([
            self.prompt_encoder.not_a_point_embed.weight,
            self.prompt_encoder.point_embeddings[0].weight,
            self.prompt_encoder.point_embeddings[1].weight,
            ]).detach()

        self.features = self.image_encoder(input)

        auto_prompt_list, class_features, final_attn_weight_list, feature_list, feature_with_pe_list = self.auto_prompt(
            image=self.features.detach(), 
            image_pe=self.get_dense_pe().detach(), 
            points_type=points_type,
            hard=False,
            bbox_masks=bbox_mask
        )

        sparse_embeddings = torch.cat((sparse_embeddings, auto_prompt_list[1]), dim=1)
        if points is not None:
            num_provided_points = points.shape[2]
            # print(num_provided_points)
            point = torch.cat([points[:,1]]+[points[:,0]],dim=1) # [B, n_points, 2]
            point_label = torch.zeros(bs,2*num_provided_points,dtype=torch.int,device=point.device)
            non_empty_mask = (point.sum(-1) > 0).flatten()
            point_label[:,:num_provided_points] = 1
            # print(point_label)
            # point = point[:,non_empty_mask,:]
            # point_label = point_label[:,non_empty_mask]
            point_embeddings = self.prompt_encoder._embed_points(point, point_label, pad=None)
            sparse_embeddings = torch.cat((sparse_embeddings, point_embeddings), dim=1)
        
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        if bbox_mask is not None:
            masks[bbox_mask[:,1:]<0.5] = float('-inf') 
        
        if return_points:
            return masks, final_attn_weight_list
        else:
            return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
        self.loss_attn = self.criterionAttn(self.final_attn_weight_list,self.gt_mask,self.feature_with_pe_list,verbose=self.verbose)
        self.loss_G += self.loss_attn

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad





class AutoPrompt(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transformer_dim: int,
        embed_dim: int = 768,
        num_class_tokens: int = 4,
        num_points: int = 4,
        use_auto_points: bool = True,
        use_classification_head: bool = True,
        use_feature_prompt: bool = True,
        num_adapt_emb: int = 8,
        temp: float = 1.,
    ):
        super().__init__()
        self.use_auto_points = use_auto_points
        self.use_classification_head = use_classification_head
        self.temp = temp
        print('use_auto_points:',self.use_auto_points)
        self.auto_prompt_encoder = ModifiedTwoWayTransformer(
            depth=4,
            embedding_dim=transformer_dim,
            mlp_dim=512,
            num_heads=8
        )
        # self.image_proj =  MLP(
        #     input_dim=transformer_dim,
        #     hidden_dim=transformer_dim * 2,
        #     output_dim=transformer_dim, 
        #     num_layers=2,)

        self.image_proj = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                transformer_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(transformer_dim),
            nn.Conv2d(
                transformer_dim,
                transformer_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(transformer_dim),
        )
        self.layer_selecter = ModifiedAttention(
            transformer_dim, 2, downsample_rate=2
        )
        self.selecter_norm = nn.LayerNorm(transformer_dim)
        

        # # init to identity
        # nn.init.eye_(self.image_proj.weight.data)
        # nn.init.zeros_(self.image_proj.bias.data)

        self.image_norm = nn.LayerNorm(transformer_dim)
        # classification head for class embedding
        if use_classification_head:
            self.classifier = ModifiedTransformer(
                depth=4,
                embedding_dim=transformer_dim,
                mlp_dim=512,
                num_heads=8
            )
            self.classifier_head = nn.Linear(transformer_dim,1)
            

        

        self.num_classes = num_classes
        self.num_class_tokens = num_class_tokens
        self.num_points = num_points

        # define the class tokens and prompt tokens
        scale = transformer_dim ** -0.5
        self.class_emb = nn.Parameter(scale * torch.randn(num_classes, num_class_tokens, transformer_dim)) # [n_classes, background] x n_class_tokens x c
        self.prompt_emb = nn.Parameter(scale * torch.randn(num_points, transformer_dim))
        self.layer_emb = nn.Parameter(scale * torch.randn(1, 1, transformer_dim))

        self.classifier_class_emb = nn.Parameter(scale * torch.randn(num_classes, transformer_dim))

        # out layer for point classification
        # self.point_loc_project = MLP(
        #     input_dim=transformer_dim,
        #     hidden_dim=transformer_dim,
        #     output_dim=transformer_dim, 
        #     num_layers=2,)
        
        self.point_loc_project_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_points)
            ]
        )
        self.point_loc_project_norm = nn.LayerNorm(transformer_dim)

        # self.point_out = MLP(
        #     input_dim=transformer_dim,
        #     hidden_dim=transformer_dim,
        #     # output_dim=3, # point type: no points, pos, neg
        #     output_dim=2, # point type: no points, yes points
        #     num_layers=2,)

        # prompt learning for the prompt encoder
        self.use_feature_prompt = use_feature_prompt
        if use_feature_prompt:
            self.point_feature_project = MLP(
                input_dim=transformer_dim,
                hidden_dim=transformer_dim,
                output_dim=transformer_dim, 
                num_layers=2,)
            self.point_feature_norm = nn.LayerNorm(transformer_dim)
        self.num_adapt_emb = num_adapt_emb
        if num_adapt_emb > 0:
            print('Num adapt emb:',num_adapt_emb)
            self.adapt_emb = nn.Parameter(scale * torch.randn(1, num_adapt_emb, transformer_dim))

    def project_and_select_feature(self, image_embedding):
        n_embs, bs, h, w, _ = image_embedding.shape
        image_embedding = image_embedding.reshape(n_embs * bs, h, w, -1).permute(0, 3, 1, 2)
        out_image_embedding = self.image_proj(image_embedding) # (n_embs * bs, c, h, w)
        out_image_embedding = out_image_embedding.reshape(n_embs, bs, -1, h, w) # (n_embs, bs, c, h, w)
        image_embedding = out_image_embedding.flatten(3).mean(-1).reshape(n_embs, bs, -1).permute(1,0,2)#(bs,n_embs,c)
        q = self.layer_emb.expand(
            bs, -1, -1
        ) # bs x 1 x c
        attn_out, attn_logits = self.layer_selecter(q=q, k=image_embedding, v=image_embedding)
        q = q + attn_out
        q = self.selecter_norm(q)
        layer_weight = torch.softmax(q @ image_embedding.permute(0,2,1),dim=-1) #(bs,1,n_embs)
        # print(layer_weight)
        layer_weight = layer_weight.permute(2,0,1)[:,:,:,None,None] #(n_embs, bs, 1, 1, 1)
        out_image_embedding = (out_image_embedding * layer_weight).sum(0)

        return out_image_embedding



    def softmax(self, x, dim=-1, hard=False):
        if hard:
            x = nn.functional.gumbel_softmax(x, tau=1, hard=True, eps=1e-10, dim=dim)
        else:
            x = nn.functional.softmax(x,dim=-1)
        return x

    def forward_single_class(self, class_ids, image, image_pe, points_type, hard=False):

        bs, c, h, w = image.shape
        # get the class emb correspond to each class
        class_emb = torch.index_select(self.class_emb, dim=0, index=class_ids) # B x N_class_tokens x C

        id_mat = torch.arange(0,self.num_classes,device = class_emb.device).unsqueeze(0).expand(
            image.size(0), -1
        ) # B x N_points x N_classes_tokens
        id_mat = id_mat[~torch.eq(id_mat,class_ids.unsqueeze(1))]

        non_class_emb = torch.index_select(self.class_emb, dim=0, index=id_mat).reshape(bs,-1,c) # B x N_class_tokens x C
        # print(non_class_emb.shape,class_ids)

        output_tokens = self.prompt_emb.unsqueeze(0).expand(
            image.size(0), -1, -1
        ) # B x N_points x C


        tokens = torch.cat((output_tokens, class_emb), dim=1)

        non_tokens = torch.cat((output_tokens, non_class_emb), dim=1)

        if self.num_adapt_emb > 0:
            # B x N_points x C
            tokens = torch.cat([tokens, self.adapt_emb.expand(bs, -1, -1)],dim=1)
        tokens, features, final_attn_weight = self.auto_prompt_encoder(
            image_embedding=image,
            image_pe=image_pe,
            point_embedding=tokens,
        )
        if self.num_adapt_emb > 0:
            # B x N_points x C
            adapt_emb = tokens[:,self.num_points+self.num_class_tokens:,:]
            tokens = tokens[:,:self.num_points+self.num_class_tokens,:]

        non_tokens, non_features, _ = self.auto_prompt_encoder(
            image_embedding=image,
            image_pe=image_pe,
            point_embedding=non_tokens,
        )

        class_features = None
        if self.use_classification_head:
            classifier_class_tokens = self.classifier_class_emb.unsqueeze(0).expand(
                        image.size(0), -1, -1
                    ) # B x N_classes x C
            class_features ,_ ,_ = self.classifier(
                    image_embedding=image,
                    image_pe=image_pe,
                    point_embedding=classifier_class_tokens,)
            class_features = self.classifier_head(class_features)

        

        if self.use_auto_points:
            pos_type = points_type[[True,False,True],:]
            neg_type = points_type[[True,True,False],:]
            # points_embeddings = self.point_out(tokens[:,:self.num_points,:])
            points_embeddings = self.softmax(points_embeddings,dim=-1,hard=hard) @ pos_type

            # non_points_embeddings = self.point_out(non_tokens[:,:self.num_points,:])
            non_points_embeddings = self.softmax(non_points_embeddings,dim=-1,hard=hard) @ neg_type

            # B x N_points x C @ B x C x N_feat -> B x N_points x N_feat
            final_attn_weight = self.point_loc_project(tokens[:,:self.num_points,:]) 
            points_feature_prompts = tokens[:,self.num_points:,:]
            final_attn_weight = final_attn_weight @ features.permute(0,2,1)
            final_attn = self.softmax(final_attn_weight,dim=-1,hard=hard) #B x N_points x N_feat
            points_loc = final_attn @ image_pe.flatten(2).permute(0, 2, 1)

            # B x N_points x C @ B x C x N_feat -> B x N_points x N_feat
            non_final_attn_weight = self.point_loc_project(non_tokens[:,:self.num_points,:]) @ non_features.permute(0,2,1)
            non_final_attn = self.softmax(non_final_attn_weight,dim=-1,hard=hard) #B x N_points x N_feat
            non_points_loc = non_final_attn @ image_pe.flatten(2).permute(0, 2, 1)

            points_embeddings = points_embeddings + points_loc
            non_points_embeddings = non_points_embeddings + non_points_loc

            points_embeddings = torch.cat([points_embeddings,non_points_embeddings],dim=1)
            if self.use_feature_prompt:
                mlp_out = self.point_feature_project(points_feature_prompts)
                points_feature_prompts = points_feature_prompts + mlp_out
                points_feature_prompts = self.point_feature_norm(points_feature_prompts)
                points_embeddings = torch.cat([points_embeddings, points_feature_prompts],dim=1)
            if self.num_adapt_emb > 0:
                 # B x N_points x C
                points_embeddings = torch.cat([points_embeddings, adapt_emb],dim=1)

            return points_embeddings, class_features, final_attn
        else:
            # B x N_points x C @ B x C x N_feat -> B x N_points x N_feat
            final_attn_weight = self.point_loc_project(tokens[:,self.num_points:,:]) 
            final_attn_weight = final_attn_weight[:,:self.num_points,:] @ features.permute(0,2,1)
            dense_embeddings = final_attn_weight[:,0,:].view(bs,-1,h,w) #B x 1 x H x W

            return dense_embeddings, class_features, final_attn_weight

    def forward(self, image, image_pe, points_type, hard=False, bbox_masks=None):

        # print(image_pe.shape)
        bs, c, h, w = image.shape
        # image = image.detach()
        if bbox_masks is not None:
            bbox_masks = F.interpolate(
                bbox_masks.float(),
                size=(h,w),
                mode="nearest-exact",
                # align_corners=True,
            ) #[B,N_classes,H,W]
            bbox_masks = bbox_masks.flatten(2) #[B,N_classes,HW]

        class_features = None
        if self.use_classification_head:
            classifier_class_tokens = self.classifier_class_emb.unsqueeze(0).expand(
                        image.size(0), -1, -1
                    ) # B x N_classes x C
            class_features ,_ ,_ = self.classifier(
                    image_embedding=image,
                    image_pe=image_pe,
                    point_embedding=classifier_class_tokens,)
            class_features = self.classifier_head(class_features)

        #     class_features = self.encode_classes(image,image_pe)



        pos_type = points_type[[True,False,True],:]
        # pos_type = points_type[[False,True,True],:]
        neg_type = points_type[[True,True,False],:]
        class_ids=torch.zeros((bs),dtype=torch.int,device=image.device)
        
        output_tokens = self.prompt_emb.unsqueeze(0).expand(
            image.size(0), -1, -1
        ) # B x N_points x C

        points_attn_list = []
        points_loc_list = []
        final_attn_list = []
        feature_prompt_list = []
        feature_list = []
        class_features_list = []
        feature_with_pe_list = []
        for i in range(self.num_classes):
            # get the class emb correspond to each class
            class_emb = torch.index_select(self.class_emb, dim=0, index=class_ids+i) # B x N_class_tokens x C

            tokens = torch.cat((output_tokens, class_emb), dim=1)

            # if self.num_adapt_emb > 0:
            #      # B x N_points x C
            #     tokens = torch.cat([tokens, self.adapt_emb.expand(bs, -1, -1)],dim=1)

            tokens, features, final_attn_weight = self.auto_prompt_encoder(
                image_embedding=image,
                image_pe=image_pe,
                point_embedding=tokens,
            )
            feature_list.append(features)

            # if self.num_adapt_emb > 0:
            #      # B x N_points x C
            #     adapt_emb = tokens[:,self.num_points+self.num_class_tokens:,:]
            #     tokens = tokens[:,:self.num_points+self.num_class_tokens,:]

            if self.use_auto_points:
                # points_embeddings = self.point_out(tokens[:,:self.num_points,:]) 
                # points_embeddings = self.softmax(points_embeddings,dim=-1,hard=hard) 
                # # remove small attn to reduce confusion
                # points_embeddings = points_embeddings * (points_embeddings > 0.001).detach()
                # points_attn_list.append(points_embeddings)
                
                # final_attn_weight = self.point_loc_project(tokens[:,:self.num_points,:])
                point_tokens_out = tokens[:,:self.num_points,:]
                final_attn_weight: List[torch.Tensor] = []
                for point_i in range(self.num_points):
                    final_attn_weight.append(
                        self.point_loc_project_mlps[point_i](point_tokens_out[:, point_i, :])
                    )
                final_attn_weight = torch.stack(final_attn_weight, dim=1) + point_tokens_out
                final_attn_weight = self.point_loc_project_norm(final_attn_weight)
                # feature_with_pe_list.append(final_attn_weight)


                points_feature_prompts = tokens[:,self.num_points:,:]

                # B x N_points X HW
                final_attn_weight = final_attn_weight @ features.permute(0,2,1)
                # feature_with_pe = (features.permute(0,2,1) + image_pe.flatten(2))
                # feature_with_pe = features.permute(0,2,1) 
                # # final_attn_weight = final_attn_weight @ feature_with_pe
                # feature_with_pe = torch.softmax(final_attn_weight,dim=-1) @ feature_with_pe.permute(0,2,1)
                # feature_with_pe_list.append(feature_with_pe)
                # print(final_attn_weight.shape)

                # higher tempure will make the point more concentrate
                final_attn_weight = final_attn_weight * self.temp
                if bbox_masks is not None:
                    # print(bbox_masks.shape)
                    bbox_mask = bbox_masks[:,i:i+1].expand(-1,self.num_points,-1) #[B,N_points,HW]
                    # print(bbox_mask.unique(),bbox_mask.sum(),final_attn_weight.max())
                    # print(torch.min(final_attn_weight, dim=-1, keepdim=True))
                    final_attn_weight[bbox_mask<0.5] = -999.
                final_attn = self.softmax(final_attn_weight,dim=-1,hard=hard) #B x N_points x N_feat
                # points_loc = final_attn @ image_pe.flatten(2).permute(0, 2, 1)
                points_loc = final_attn @ (image_pe+image).flatten(2).permute(0, 2, 1)
                feature_with_pe_list.append(points_loc)
                # points_loc = final_attn @ image_pe.flatten(2).permute(0, 2, 1)
                # points_loc = feature_with_pe
                # points_loc = (final_attn  * (final_attn > 0.01).detach()) @ feature_with_pe.permute(0, 2, 1) # B x HW x C
                # gather the image features using the points and determine if the point should be considered as positive
                # points_embeddings = self.point_out(final_attn @ image.flatten(2).permute(0, 2, 1)) 
                # points_embeddings = self.point_out(tokens[:,:self.num_points,:]) 
                # class_features_list.append(points_embeddings)
                # points_embeddings = torch.softmax(points_embeddings,dim=-1) 
                # points_attn_list.append(points_embeddings)
                # remove small attn to reduce confusion
                # points_embeddings = points_embeddings * (points_embeddings > 0.01).detach()
                
                points_loc_list.append(points_loc)
                
                # scale the attention by the classification result of the image
                # if self.use_classification_head:
                # final_attn_list.append(torch.softmax(final_attn_weight,dim=-1) * points_embeddings[:,:,1:])
                # else:
                final_attn_list.append(torch.softmax(final_attn_weight,dim=-1))
                
                if self.use_feature_prompt:
                    mlp_out = self.point_feature_project(points_feature_prompts)
                    points_feature_prompts = points_feature_prompts + mlp_out
                    points_feature_prompts = self.point_feature_norm(points_feature_prompts)
                    feature_prompt_list.append(points_feature_prompts)
                    # points_embeddings = torch.cat([points_embeddings, points_feature_prompts],dim=1)
        # use other points as neg_points for this class
        points_embeddings_list = []
        for i in range(len(points_loc_list)):
            # always use the points to prompt SAM but filter out the mask later.
            point = torch.cat([pos_type[1:] + points_loc_list[i]]+[(neg_type[1:] + points_loc_list[j]) for j in range(len(points_loc_list)) if i!=j],dim=1)
            # point = torch.cat([points_attn_list[i] @ pos_type + points_loc_list[i]]+[(points_attn_list[j] @ neg_type + points_loc_list[j]) for j in range(len(points_loc_list)) if i!=j],dim=1)
            if self.use_feature_prompt:
                point = torch.cat([point, feature_prompt_list[i]],dim=1)
            if self.num_adapt_emb > 0:
                 # B x N_points x C
                point = torch.cat([point, self.adapt_emb.expand(bs,-1,-1)],dim=1)
            points_embeddings_list.append(point)

        # B x N_class x N_points x HW  -> B x N_class x N_points x H x W
        final_attn_list = torch.stack(final_attn_list,dim=1).reshape(bs,-1,self.num_points,h, w)

        # B x N_class x N_points x 2 -> B x N_class x N_points x 1 
        # class_features = torch.stack(class_features_list,dim=1)

        feature_with_pe_list = torch.stack(feature_with_pe_list,dim=1).reshape(bs,-1,self.num_points,c)
        # print(final_attn_list.shape)
        return points_embeddings_list, class_features, final_attn_list, feature_list, feature_with_pe_list

    # def forward(self, class_ids, image, image_pe, points_type):

    #     bs, c, h, w = image.shape
    #     # get the class emb correspond to each class
    #     class_emb = torch.index_select(self.class_emb, dim=0, index=class_ids) # B x N_class_tokens x C
    #     output_tokens = self.prompt_emb.unsqueeze(0).expand(
    #         image.size(0), -1, -1
    #     ) # B x N_points x C

    #     tokens = torch.cat((output_tokens, class_emb), dim=1)

    #     tokens, features, final_attn_weight = self.auto_prompt_encoder(
    #         image_embedding=image,
    #         image_pe=image_pe,
    #         point_embedding=tokens,
    #     )

    #     class_features = None
    #     if self.use_classification_head:
    #         class_features = self.encode_classes(image,image_pe)

    #     if self.use_auto_points:
    #         points_embeddings = self.point_out(tokens[:,:self.num_points,:])
    #         points_embeddings = nn.functional.softmax(points_embeddings,dim=-1) @ points_type

    #         # B x N_points x C @ B x C x N_feat -> B x N_points x N_feat
    #         final_attn_weight = self.point_loc_project(tokens[:,:self.num_points,:]) @ features.permute(0,2,1)
    #         final_attn = nn.functional.softmax(final_attn_weight,dim=-1) #B x N_points x N_feat
    #         points_loc = final_attn @ image_pe.flatten(2).permute(0, 2, 1)

    #         points_embeddings = points_embeddings + points_loc

    #         return points_embeddings, class_features, final_attn
    #     else:
    #         # B x N_points x C @ B x C x N_feat -> B x N_points x N_feat
    #         final_attn_weight = self.point_loc_project(tokens[:,:self.num_points,:]) @ features.permute(0,2,1)
    #         dense_embeddings = final_attn_weight[:,0,:].view(bs,-1,h,w) #B x 1 x H x W

    #         return dense_embeddings, class_features, final_attn_weight

    def encode_classes(self,image,image_pe):
        bs, c, h, w = image.shape
        image = image.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1) # B x HW x C

        n_classes, n_tokens, _ = self.class_emb.shape
        class_tokens = self.class_emb.mean(1)  # N_class x N_tokens x C -> N_class x C

        output_tokens = class_tokens.unsqueeze(0).expand(
            bs, -1, -1
        ) # B x  N_class x C

        k = image + image_pe
        attn_out, attn_logits = self.class_encoder(q=output_tokens, k=k, v=image)

        output_tokens = output_tokens + attn_out
        output_tokens = self.norm_class(output_tokens) # B x  N_class x C

        # MLP block
        output_tokens = self.class_head(output_tokens) # B x  N_class x 1

        return output_tokens