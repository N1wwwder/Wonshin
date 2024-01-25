# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:31:28 2024

@author: 11693
"""

import numpy as np
import torch
from torch.utils import data  # 从torch的utils里面调用一些处理数据的模块
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 目的是实现小批量随机梯度下降法中的小批量的选取，实现对数据集按照batch_size进行分批，并随机进行读取。load_array
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # *号作用是对data_arrays解开入参，这一行等价于下面一行的代码
   #dataset = data.TensorDataset(data_arrays[0],data_arrays[1])
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 从数据集中随机取batch_size个数据

next(iter(data_iter))

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))  # Sequential: list of layers 一个容器
# Linear参数意义为输入维度和输出维度

net[0].weight.data.normal_(0, 0.01)  # 等价于参数初始化
# weight访问w，data是真实data，mormal_是使用正态分布替换data的值
net[0].bias.data.fill_(0)  #bias偏差，

loss = nn.MSELoss()  # pytorch自带均方根误差函数

trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 小批量梯度下降
# net.parameters包括了net里传入的所有参数，lr相同的是学习率η

# 代码训练过程与前代码比较相像
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)   # 计算总损失
        trainer.zero_grad()   # 梯度设为0
        l.backward()          # 计算梯度
        trainer.step()        # 权值参数更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)