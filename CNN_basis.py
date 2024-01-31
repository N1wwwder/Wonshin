# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:06:47 2024

@author: 11693
"""

import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape  # 找到行数和列数
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i+h, j : j+w]*K).sum()
    return Y

"""下面实现二维卷积层"""
class Conv2D(nn.Module):
    def _init_(self, kernel_size):
        super()._init_()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """向前传播函数"""
        return corr2d(x, self.weight) + self.bias

    