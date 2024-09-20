# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        # import pdb
        # pdb.set_trace() 
        out = []
        for xx in split_x:
            # pdb.set_trace()
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            out.append(h)
        return torch.stack(out)

    def regroup(self,x, record_len):
    # def regroup(self,input):
        # x = input[0]
        # record_len = input[1]
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    

class AttFusion_vit(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion_vit, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        # import pdb
        # pdb.set_trace() 
        out = []
        for xx in split_x:
            # pdb.set_trace()
            cav_num = xx.shape[0]
            # xx = xx.view(C,-1,cav_num)
            xx = xx.view(C,cav_num,-1)
            # pdb.set_trace()
            h = self.att(xx, xx, xx)
            h = h.view(cav_num, C, W, H)[0, ...]
            out.append(h)
        return torch.stack(out)

    def regroup(self,x, record_len):
    # def regroup(self,input):
        # x = input[0]
        # record_len = input[1]
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
