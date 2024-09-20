# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO:jinlong
from .mamba_V2 import CoVSSBlock

# from .mamba_V3 import CoVSSBlock


from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


import pdb

class MambaFusion(nn.Module):
    def __init__(self, feature_dim):
        super(MambaFusion, self).__init__()
        # self.att = ScaledDotProductAttention(feature_dim)

        self.vss = CoVSSBlock(in_chans=256,hidden_dim=256,patch_size=1)


        self.process = nn.Sequential(
            Rearrange('b h w c -> b c h w')
        )
        self.fusion_mean = nn.Sequential(
            Reduce('m c h w -> c h w', 'mean')
            )
        self.fusion_max = nn.Sequential(
            # Rearrange('(b m) c h w -> b m c h w', m=5),
            Reduce('m c h w -> c h w', 'max')
            )

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        out = []

        for xx in split_x:

            cav_num = xx.shape[0]

            #------>   N, H*W  mamba_mean_max
            xx = xx.view(cav_num, C, -1).permute(1, 0, 2)
            xx = xx.unsqueeze(0).contiguous()
            xx = self.vss(xx)
            xx = self.process(xx)
            h = xx.squeeze(0)
            xx = h.permute(1, 0, 2).view(cav_num, C, W, H).contiguous()
            fused = self.fusion_max(xx) + self.fusion_mean(xx)


            out.append(fused)
        return torch.stack(out)
    

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    



def split_features(x, num_splits=8):
    # x为输入特征，大小为[B*H*W, C, N, 1]
    B, C, W, H = x.shape
    # 划分子特征
    split_size = B // num_splits
    return [x[i*split_size:(i+1)*split_size] for i in range(num_splits)]



