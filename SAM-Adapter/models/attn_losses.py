import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnMaskLoss(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma: float = 1., temp: float = 7, include_background: bool = True, reduction='mean'):
        super(AttnMaskLoss, self).__init__()

        self.gamma = gamma
        self.reduction=reduction
        self.include_background = include_background
        self.temp = temp

    def forward(self, attn_map, target, image_pe=None,verbose=False):
        '''
        "attn_map" dimensions: - (batch_size,n_class,n_points,h1,w1)
        "target" dimensions: - (batch_size,n_class,n_c,h2,w2)
        "image_pe" dimensions: - (batch_size,n_class,c,h1,w1)
        '''


        use_var_loss = True
        # if not self.include_background:
        #     attn_map = attn_map[:,1:]
        #     target = target[:,1:]

        bs,n_class,n_points,h,w = attn_map.shape
        _,n_class_2,h2,w2 = target.shape
        # print(attn_map.shape,target.shape)

        # resize the target to the input size and repeat the n_points times
        target = F.interpolate(
                target.reshape(bs*n_class_2,1,h2,w2).float(),
                size=(h,w),
                mode="bilinear",
                align_corners=True,
            ).reshape(bs,n_class_2,1,h,w).expand(-1,-1,n_points,-1,-1) # (batch_size,n_class,n_points,h1,w1)
        if n_class_2 == 1: # if only one class
            target = torch.cat([1.-target,target],dim=1)
        # print(target.shape)

        # mask out the loss if gt does not exist
        gt_exist = target.flatten(3).sum(dim=-1) > 0 # (batch_size,n_class,n_points)
        total_attn = attn_map.flatten(3).sum(dim=-1)#.detach() # (batch_size,n_class,n_points)
        total_attn = total_attn * gt_exist
        # max_attn = attn_map.flatten(3).max(dim=-1).values # (batch_size,n_class,n_points)
        correct_attn = (attn_map * target).flatten(3).sum(dim=-1) # (batch_size,n_class,n_points)
        correct_attn = correct_attn * gt_exist
        correct_max_attn = (attn_map * target).flatten(3).max(dim=-1).values
        correct_max_attn = correct_max_attn * gt_exist
        # print(total_attn.shape,max_attn.shape,correct_attn.shape)


        # 
        percision_loss = 1.- (correct_attn + self.gamma)/(total_attn + self.gamma)
        # sharpness_loss = 1.- (max_attn * gt_exist + self.gamma)/(total_attn * gt_exist + self.gamma)
        correct_sharpness_loss = 1.- (correct_max_attn + self.gamma)/(correct_attn.detach() + self.gamma)
        loss = 2.0 * percision_loss + correct_sharpness_loss


        # points diversity loss
        temp = self.temp
        # temp = 7. * target.flatten(3).sum(dim=-1).reshape(bs*n_class,n_points,-1)  # (batch_size,n_class,n_points)
        if image_pe is not None:
            #attn_map (batch_size,n_class,n_points,h1,w1) image_pe" dimensions: - 
            class_loc_embeddings = image_pe.permute(0,2,1,3).reshape(bs*n_points,n_class,-1)
            class_loc_embeddings = F.normalize(class_loc_embeddings, dim=-1,eps=1e-8)
            # print(class_loc_embeddings.shape,image_pe.shape)
            class_loc_logits = class_loc_embeddings @ class_loc_embeddings.permute(0,2,1)
            # print(loc_logits)
            class_loc_label = torch.eye(class_loc_logits.shape[1],device=class_loc_embeddings.device).unsqueeze(0).expand(class_loc_logits.shape[0],-1,-1)
            # print(loc_label)
            class_loc_loss = F.cross_entropy(class_loc_logits * temp,class_loc_label)

            loc_embeddings = image_pe.reshape(bs*n_class,n_points,-1)
            loc_embeddings = F.normalize(loc_embeddings, dim=-1,eps=1e-8)
            # print(loc_embeddings.shape)
            loc_logits = loc_embeddings @ loc_embeddings.permute(0,2,1)
            # print(loc_logits)
            loc_label = torch.eye(loc_logits.shape[1],device=loc_embeddings.device).unsqueeze(0).expand(loc_logits.shape[0],-1,-1)
            # print(loc_label)
            loc_loss = F.cross_entropy(loc_logits * temp,loc_label)
        else:
            class_loc_loss = 0
            loc_loss = 0
            # print(loc_loss)
        # print(attn_map.reshape(bs*n_class,n_points,-1).sum(-2))
        if verbose:
            print('percision_loss:',percision_loss.sum().item() / gt_exist.sum().item(),'correct_sharpness:',correct_sharpness_loss.sum().item() / gt_exist.sum().item(),'loc_loss:',loc_loss.mean().item())
            print('corrent_attn:',correct_attn.sum().item() / gt_exist.sum().item(),'correct_max_attn:',correct_max_attn.sum().item() / gt_exist.sum().item(),'total_attn:',total_attn.sum().item() / gt_exist.sum().item())

        if self.reduction == 'mean':
            loss = loss.sum() / gt_exist.sum()
        else:
            loss = loss.sum()
        return loss + 0.2*loc_loss + 0.5*class_loc_loss
        # return loc_loss