# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:50:13 2024

@author: 11693
"""

import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#Pytorch不会隐式的去调整输入的形状
#因此我们定义了展平层（flatten）在线形层之前调整网络输入的类型
net = nn.Sequential(nn.Flatten(),nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)  # 权重的初始化非常重要，因为不恰当的初始化可能导致训练过程中出现梯度消失或爆炸的问题
        
net.apply(init_weights)  # 这里apply方法给net中的每一层都调用上述的初始化函数

loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.SGD(net.parameters(), lr = 0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)