# -*- coding: utf-8 -*-
# This code is modified from SAM.


import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from segment_anything.modeling.common import MLPBlock, LayerNorm2d



class ModifiedTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                ModifiedAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate
                )
            )

        self.final_attn_token_to_image = ModifiedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # image_embedding = image_embedding + image_pe

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        attn_logits_list = []
        # Apply transformer blocks and final layernorm
        size = 64,64
        for layer in self.layers:
            queries, keys, attn_logits, size = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_embedding,
                size=size
            )
            attn_logits_list.append(attn_logits)

        # Apply the final attention layer from the points to the image
        # q = queries + point_embedding
        # k = keys + image_embedding
        attn_out, attn_logits = self.final_attn_token_to_image(q=queries, k=keys, v=keys)
        attn_logits_list.append(attn_logits)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        # final_attn_logits = torch.mean(torch.stack(attn_logits_list), dim=0) # B x N_q x N_k
        # final_attn_weight = nn.functional.softmax(final_attn_logits) 
        final_attn_logits = attn_logits
        final_attn_weight = nn.functional.softmax(final_attn_logits,dim=-1) 

        return queries, keys, final_attn_weight


class ModifiedAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()

        self.cross_attn_token_to_image = ModifiedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=1,
                bias=True,
            ),
        )
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=False, divisor_override=None)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # self.cross_attn_image_to_token = ModifiedAttention(
        #     embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        # )
        self.norm3 = nn.LayerNorm(embedding_dim)


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, size: Tuple = (64,64)
    ) -> Tuple[Tensor, Tensor]:
        # Cross attention block, tokens attending to image embedding

        # q = queries + query_pe
        # k = keys + key_pe
        attn_out, attn_logits = self.cross_attn_token_to_image(q=queries, k=keys, v=keys)

        queries = queries + attn_out
        queries = self.norm1(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm2(queries)

        # B x N_image_tokens x C
        B, N, C = keys.shape
        keys = keys.permute(0,2,1).reshape(B,C,*size)
        mlp_out = self.conv_mlp(keys) #(bs, c, h, w)
        keys = keys + mlp_out
        keys = self.pool(keys)
        _,_, h,w = keys.shape
        # print(h,w)
        keys = keys.reshape(B,C,h*w).permute(0,2,1)
        keys = self.norm3(keys)

        # Cross attention block, image embedding attending to tokens
        # q = queries + query_pe
        # k = keys + key_pe
        # attn_out, _ = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        # keys = keys + attn_out
        # keys = self.norm3(keys)

        return queries, keys, attn_logits, (h,w)


class ModifiedTwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                ModifiedTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = ModifiedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # image_embedding = image_embedding + image_pe

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        attn_logits_list = []
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, attn_logits = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_embedding
            )
            attn_logits_list.append(attn_logits)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_embedding
        attn_out, attn_logits = self.final_attn_token_to_image(q=q, k=k, v=keys)
        attn_logits_list.append(attn_logits)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        # final_attn_logits = torch.mean(torch.stack(attn_logits_list), dim=0) # B x N_q x N_k
        # final_attn_weight = nn.functional.softmax(final_attn_logits) 
        final_attn_logits = attn_logits
        final_attn_weight = nn.functional.softmax(final_attn_logits,dim=-1) 

        return queries, keys, final_attn_weight

class ModifiedTwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = ModifiedAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = ModifiedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(
                embedding_dim,
                embedding_dim,
                kernel_size=1,
                bias=True,
            ),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = ModifiedAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm5 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, size: Tuple = (64,64),
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries, _ = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out, _ = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)


        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out, attn_logits = self.cross_attn_token_to_image(q=q, k=k, v=keys)

        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # B x N_image_tokens x C
        B, N, C = keys.shape
        keys = keys.permute(0,2,1).reshape(B,C,*size)
        mlp_out = self.conv_mlp(keys) #(bs, c, h, w)
        keys = keys + mlp_out
        keys = keys.reshape(B,C,N).permute(0,2,1)
        keys = self.norm4(keys)

        # # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm5(keys)

        return queries, keys, attn_logits


class ModifiedAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    # def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    #     # Input projections
    #     q = self.q_proj(q)
    #     k = self.k_proj(k)
    #     v = self.v_proj(v)

    #     # Separate into heads
    #     q = self._separate_heads(q, self.num_heads)
    #     k = self._separate_heads(k, self.num_heads)
    #     v = self._separate_heads(v, self.num_heads)

    #     # Attention
    #     _, _, _, c_per_head = q.shape
    #     attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
    #     attn = attn / math.sqrt(c_per_head)
    #     attn = torch.softmax(attn, dim=-1)

    #     # Get output
    #     out = attn @ v
    #     out = self._recombine_heads(out)
    #     out = self.out_proj(out)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, average_attn_weights=True) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        logits = attn
        attn = torch.softmax(attn, dim=-1)
        # attn = nn.functional.gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=- 1)
        if average_attn_weights:
            attn_weights = logits.mean(1)
        else:
            attn_weights = logits
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn_weights

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation = nn.functional.gelu,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.activation = activation
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
