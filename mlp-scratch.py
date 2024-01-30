# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:13:53 2024

@author: 11693
"""

import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 输入层
W1 = torch.randn(num_inputs, num_hiddens, requires_grad=True)  # 行，列，要更新梯度
b1 = torch.zeros(num_hiddens, requires_grad=True)
# 输出层
W2 = torch.randn(num_hiddens, num_outputs, requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2]  # 全部的参数

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 +b1)
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr = lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.predict_ch3(net, test_iter)