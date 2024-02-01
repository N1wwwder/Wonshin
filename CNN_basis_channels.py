# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:30:40 2024

@author: 11693
"""

import torch
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    """多输入通道互相关运算"""
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    """多通道输出的互相关运算"""
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
# 这里的K是4D的，X是3D的，偏移是2D的



