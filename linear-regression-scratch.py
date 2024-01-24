# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:14:26 2024

@author: 11693
"""

import torch
import random
from d2l import torch as d2l

#设计样本生成函数
def synthetic_data(w, b, num_examples):  # 生成噪声
    X = torch.normal(0, 0.1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

#获取噪声散点图
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('feature: ', features[0], 'labels: ', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
# features[:, 1].detach().numpy()  横坐标，detach()将features分离出计算图，numpy转换numpy数组，1：散点大小

# 设计迭代器：接受批量大小、特征矩阵和标签向量，生成大小为batchsize的小批量数据
# 通过这个迭代器我以读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))  # 生成从0 —— num_examples - 1的list
    random.shuffle(indices)  # 打乱列表的顺序
    for i in range(0, num_examples, batch_size):  # 以batch_size为步长，从0遍历到num_examples - 1
        batch_indices = torch.tensor(indices[i: min(num_examples, i + batch_size)])  #每一次遍历从i开始取batch_size个索引，min函数保证索引不超过最大值
        yield features[batch_indices], labels[batch_indices]     
#yield的生成迭代器办法可以帮助节省内存
#yield是一个定义生成器的关键字，生成器是一种特殊的函数，逐个产生一系列的值并返回
#而函数会保留当前转态，下次调用迭代器时，又会从当前yield处开始运行，返回下一批新的batch个数据
#使用迭代器的办法用for循环
#for x,y in data_iter(batch_size,features,labels)  就会迭代循环调用这个迭代器直到迭代运行结束
batch_size = 10
for X,y in data_iter(batch_size, features, labels):
    print('X', X, '\ny:', y)
    break  # 迭代运行一次就退出
    
#训练
def linreg(X, w, b):
    return torch.matmul(X, w) + b
def square_loss(y, y_hat):
    return (y_hat - y.reshape(y_hat.shape))**2/2  # 要保证y和y_hat的形状一致
#随机梯度下降模型
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

w = torch.normal(0, 0.01, size = (2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad = True) 

lr = 0.03
num_epochs = 3
net = linreg
loss = square_loss

for epoch in range ( num_epochs ):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 计算网络输出和y的损失
        l.sum().backward()  #更新梯度
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

        
    