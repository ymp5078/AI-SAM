# -*- coding: utf-8 -*-
# This code is modified from SAM.

import math
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .modified_transformer import ModifiedTwoWayTransformer, MLP, ModifiedAttention, ModifiedTransformer, LayerNorm2d
from segment_anything import sam_model_registry
from .sam_lora_image_encoder import LoRA_Sam


class AISAM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sam_checkpoint: str,
        sam_model: str = 'vit_h',
        num_class_tokens: int = 4,
        num_points: int = 4,
        use_classification_head: bool = True,
        use_hard_points: bool = False,
        use_lora: bool = False
    ):
        super().__init__()
        if use_lora:
            self.sam = LoRA_Sam(sam_model_registry[sam_model](checkpoint=sam_checkpoint))
            self.image_encoder = self.sam.sam.image_encoder
            self.mask_decoder = self.sam.sam.mask_decoder
            self.prompt_encoder = self.sam.sam.prompt_encoder
        else:
            self.sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint)
            self.image_encoder = self.sam.image_encoder
            self.mask_decoder = self.sam.mask_decoder
            self.prompt_encoder = self.sam.prompt_encoder
        self.hard = use_hard_points
        print('hard:',use_hard_points)
        
        transformer_dim = self.mask_decoder.transformer_dim
        self.num_classes=num_classes
        dim_dict = {'vit_b':768,'vit_l':1024,'vit_h':1280,'vit_t':320,}
        encoder_embed_dim = dim_dict[sam_model]
        self.auto_prompt = AutoPrompt(
            num_classes=num_classes,
            transformer_dim=transformer_dim,
            embed_dim = encoder_embed_dim,
            num_class_tokens=num_class_tokens,
            num_points=num_points,
            use_classification_head=use_classification_head,
        )

        # freeze image encoder
        if not use_lora:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
    
    def forward(self, image, bbox_masks=None, points=None, low_res = False):
        # automatic forward
        image_embedding, all_layer_image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64), (L, B, 256, 64, 64)
        image_pe = self.prompt_encoder.get_dense_pe() # (B, 256, 64, 64)
        bs, c, h, w = image_embedding.shape

        ori_res_masks_list = []
        points_type = torch.cat([
            self.prompt_encoder.not_a_point_embed.weight,
            self.prompt_encoder.point_embeddings[0].weight,
            self.prompt_encoder.point_embeddings[1].weight,
            ]).detach()
        
        if points is not None: # [B, n_class, n_points, 2]
            num_provided_points = points.shape[2]
            point_list = []
            point_label_list = []
            for i in range(self.num_classes):
                # always use the points to prompt SAM but filter out the mask later.
                assert bs == 1, f'current only support bs==1 not {bs}'
                point = torch.cat([points[:,i]]+[points[:,j] for j in range(self.num_classes) if i!=j],dim=1) # [B, n_points, 2]
                point_label = torch.zeros(bs,self.num_classes*num_provided_points,dtype=torch.int,device=point.device)
                non_empty_mask = (point.sum(-1) > 0).flatten()
                point_label[:,:num_provided_points] = 1
                point = point[:,non_empty_mask,:]
                point_label = point_label[:,non_empty_mask]
                point_list.append(point)
                point_label_list.append(point_label)
        auto_prompt_list, class_features, final_attn_weight_list, feature_list, feature_with_pe_list = self.auto_prompt(
            image=image_embedding.detach(), 
            image_pe=image_pe.detach(), 
            points_type=points_type,
            hard=self.hard,
            bbox_masks=bbox_masks # [B,N_class,H,W]
        )

        for i in range(self.num_classes):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_list[i],point_label_list[i]) if points is not None else None,
                auto_prompt=auto_prompt_list[i],# if points is None else None,
                boxes=None,
                masks=None,
            )
            
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=image_pe,  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            if low_res:
                ori_res_masks = low_res_masks
            else:
                ori_res_masks = F.interpolate(
                    low_res_masks,
                    size=(image.shape[2], image.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            ori_res_masks_list.append(ori_res_masks)

        ori_res_masks = torch.cat(ori_res_masks_list,dim=1)
        if bbox_masks is not None:
            bbox_masks = F.interpolate(
                        bbox_masks.float(),
                        size=(ori_res_masks.shape[2], ori_res_masks.shape[3]),
                        mode="nearest-exact",
                    )

            ori_res_masks[bbox_masks<0.5] = float('-inf') 
        return ori_res_masks, class_features, final_attn_weight_list, feature_with_pe_list

    def forward_single_class(self, image, class_ids, bbox_masks=None, points=None, low_res = False):
        assert image.shape[0] == 1, f'only support batch=1 but input batch={image.shape[0]}'
        # interactive forward
        image_embedding, all_layer_image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64), (L, B, 256, 64, 64)
        image_pe = self.prompt_encoder.get_dense_pe() # (B, 256, 64, 64)
        bs, c, h, w = image_embedding.shape
        

        ori_res_masks_list = []
        points_type = torch.cat([
            self.prompt_encoder.not_a_point_embed.weight,
            self.prompt_encoder.point_embeddings[0].weight,
            self.prompt_encoder.point_embeddings[1].weight,
            ]).detach()
        
        if points is not None: # [B, n_class, n_points, 2]
            num_provided_points = points.shape[2]
            point_list = []
            point_label_list = []
            for i in range(self.num_classes):
                # always use the points to prompt SAM but filter out the mask later.
                point = torch.cat([points[:,i]]+[points[:,j] for j in range(self.num_classes) if i!=j],dim=1) # [B, n_points, 2]
                point_label = torch.zeros(bs,self.num_classes*num_provided_points,dtype=torch.int,device=point.device)
                non_empty_mask = (point.sum(-1) > 0).flatten()
                point_label[:,:num_provided_points] = 1
                point = point[:,non_empty_mask,:]
                point_label = point_label[:,non_empty_mask]
                point_list.append(point)
                point_label_list.append(point_label)
        auto_prompt_list, class_features, final_attn_weight_list, feature_list, feature_with_pe_list = self.auto_prompt(
            image=image_embedding.detach(), 
            image_pe=image_pe.detach(), 
            points_type=points_type,
            hard=self.hard,
            bbox_masks=bbox_masks # [B,N_class,H,W]
        )

        for i in range(self.num_classes):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_list[i],point_label_list[i]) if points is not None else None,
                auto_prompt=auto_prompt_list[i],# if points is None else None,
                boxes=None,
                masks=None,
            )
            
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=image_pe,  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            if low_res:
                ori_res_masks = low_res_masks
            else:
                ori_res_masks = F.interpolate(
                    low_res_masks,
                    size=(image.shape[2], image.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            ori_res_masks_list.append(ori_res_masks)

        ori_res_masks = torch.cat(ori_res_masks_list,dim=1)
        if bbox_masks is not None:
            bbox_masks = F.interpolate(
                        bbox_masks.float(),
                        size=(ori_res_masks.shape[2], ori_res_masks.shape[3]),
                        mode="nearest-exact",
                    )

            ori_res_masks[bbox_masks<0.5] = float('-inf') 
        return ori_res_masks, class_features, final_attn_weight_list, feature_with_pe_list



class AutoPrompt(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transformer_dim: int,
        embed_dim: int = 768,
        num_class_tokens: int = 4,
        num_points: int = 4,
        use_classification_head: bool = True,
    ):
        super().__init__()
        self.use_classification_head = use_classification_head
        self.auto_prompt_encoder = ModifiedTwoWayTransformer(
            depth=4,
            embedding_dim=transformer_dim,
            mlp_dim=512,
            num_heads=8
        )

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
        
        self.point_loc_project_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_points)
            ]
        )
        self.point_loc_project_norm = nn.LayerNorm(transformer_dim)

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

    def random_gt_points(self,masks,n_ponints):
        """
            masks: [bs, n_class, h, w] one hot labels
        """
        masks = masks.flatten(2) # [bs, n_class, h, w]
        n_tokens = masks.shape[-1]

    def forward(self, image, image_pe, points_type, hard=False, bbox_masks=None):

        bs, c, h, w = image.shape

        # prepare the bbox_masks for the output weights
        if bbox_masks is not None:
            bbox_masks = F.interpolate(
                bbox_masks.float(),
                size=(h,w),
                mode="nearest-exact",
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



        pos_type = points_type[[True,False,True],:]
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


            tokens, features, final_attn_weight = self.auto_prompt_encoder(
                image_embedding=image,
                image_pe=image_pe,
                point_embedding=tokens,
            )
            feature_list.append(features)

            point_tokens_out = tokens[:,:self.num_points,:]
            final_attn_weight: List[torch.Tensor] = []
            for point_i in range(self.num_points):
                final_attn_weight.append(
                    self.point_loc_project_mlps[point_i](point_tokens_out[:, point_i, :])
                )
            final_attn_weight = torch.stack(final_attn_weight, dim=1) + point_tokens_out
            final_attn_weight = self.point_loc_project_norm(final_attn_weight)


            points_feature_prompts = tokens[:,self.num_points:,:]

            # B x N_points X HW
            final_attn_weight = final_attn_weight @ features.permute(0,2,1)

            # constrain the final_attn_weight if the input bbox_masks is avaliable
            if bbox_masks is not None:
                bbox_mask = bbox_masks[:,i:i+1].expand(-1,self.num_points,-1) #[B,N_points,HW]
                final_attn_weight[bbox_mask<0.5] = -999.

            final_attn = self.softmax(final_attn_weight,dim=-1,hard=hard) 
            points_loc = final_attn @ (image_pe+image).flatten(2).permute(0, 2, 1)
            feature_with_pe_list.append(points_loc)
            points_loc = final_attn @ image_pe.flatten(2).permute(0, 2, 1) 
            
            points_loc_list.append(points_loc)
            final_attn_list.append(torch.softmax(final_attn_weight,dim=-1))
                
        # use other points as neg_points for this class
        points_embeddings_list = []
        for i in range(len(points_loc_list)):
            # always use the points to prompt SAM but filter out the mask later.
            point = torch.cat([pos_type[1:] + points_loc_list[i]]+[(neg_type[1:] + points_loc_list[j]) for j in range(len(points_loc_list)) if i!=j],dim=1)
            
            points_embeddings_list.append(point)

        # B x N_class x N_points x HW  -> B x N_class x N_points x H x W
        final_attn_list = torch.stack(final_attn_list,dim=1).reshape(bs,-1,self.num_points,h, w)

        feature_with_pe_list = torch.stack(feature_with_pe_list,dim=1).reshape(bs,-1,self.num_points,c)
        return points_embeddings_list, class_features, final_attn_list, feature_list, feature_with_pe_list

    def forward_single_class(self, class_id, image, image_pe, points_type, hard=False, bbox_masks=None):
        
        assert image.shape[0] == 1, f'only support batch=1 but input batch={image.shape[0]}'

        bs, c, h, w = image.shape

        # prepare the bbox_masks for the output weights
        if bbox_masks is not None:
            bbox_masks = F.interpolate(
                bbox_masks.float(),
                size=(h,w),
                mode="nearest-exact",
            ) #[B,N_classes,H,W]
            bbox_masks = bbox_masks.flatten(2) #[B,N_classes,HW]
            



        pos_type = points_type[[True,False,True],:]
        neg_type = points_type[[True,True,False],:]
        class_ids=torch.zeros((bs),dtype=torch.int,device=image.device)
        
        output_tokens = self.prompt_emb.unsqueeze(0).expand(
            image.size(0), -1, -1
        ) # B x N_points x C
        class_features = None
        points_attn_list = []
        points_loc_list = []
        final_attn_list = []
        feature_prompt_list = []
        feature_list = []
        feature_with_pe_list = []
        for i in range(self.num_classes):
            # get the class emb correspond to each class
            class_emb = torch.index_select(self.class_emb, dim=0, index=class_ids+i) # B x N_class_tokens x C
            tokens = torch.cat((output_tokens, class_emb), dim=1)
            tokens, features, final_attn_weight = self.auto_prompt_encoder(
                image_embedding=image,
                image_pe=image_pe,
                point_embedding=tokens,
            )
            feature_list.append(features)

            point_tokens_out = tokens[:,:self.num_points,:]
            final_attn_weight: List[torch.Tensor] = []
            for point_i in range(self.num_points):
                final_attn_weight.append(
                    self.point_loc_project_mlps[point_i](point_tokens_out[:, point_i, :])
                )
            final_attn_weight = torch.stack(final_attn_weight, dim=1) + point_tokens_out
            final_attn_weight = self.point_loc_project_norm(final_attn_weight)


            points_feature_prompts = tokens[:,self.num_points:,:]

            # B x N_points X HW
            final_attn_weight = final_attn_weight @ features.permute(0,2,1)

            # constrain the final_attn_weight if the input bbox_masks is avaliable
            if bbox_masks is not None:
                bbox_mask = bbox_masks[:,i:i+1].expand(-1,self.num_points,-1) #[B,N_points,HW]
                final_attn_weight[bbox_mask<0.5] = -999.

            final_attn = self.softmax(final_attn_weight,dim=-1,hard=hard) 
            points_loc = final_attn @ (image_pe+image).flatten(2).permute(0, 2, 1)
            feature_with_pe_list.append(points_loc)
            points_loc = final_attn @ image_pe.flatten(2).permute(0, 2, 1) 
            
            points_loc_list.append(points_loc)
            final_attn_list.append(torch.softmax(final_attn_weight,dim=-1))
                
        # use other points as neg_points for this class
        points_embeddings_list = []
        for i in range(len(points_loc_list)):
            # always use the points to prompt SAM but filter out the mask later.
            point = torch.cat([pos_type[1:] + points_loc_list[i]]+[(neg_type[1:] + points_loc_list[j]) for j in range(len(points_loc_list)) if i!=j],dim=1)
            
            points_embeddings_list.append(point)

        # B x N_class x N_points x HW  -> B x N_class x N_points x H x W
        final_attn_list = torch.stack(final_attn_list,dim=1).reshape(bs,-1,self.num_points,h, w)

        feature_with_pe_list = torch.stack(feature_with_pe_list,dim=1).reshape(bs,-1,self.num_points,c)
        return points_embeddings_list, class_features, final_attn_list, feature_list, feature_with_pe_list

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