# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:21:53 2024

@author: 11693
"""

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from d2l import torch as d2l

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='./data',
    train = True,
    transform = trans,
    download = True,
)  # 这里的transform = trans是说明我们操作中要拿的是tensor而不是图片
mnist_test = torchvision.datasets.FashionMNIST(
    root='./data',
    train = False,
    transform = trans,
    download = True,
)  # 这里是测试数据集，用来测试效果的好坏，因此其train对应的为False。

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return text_labels[int(labels)]

fig, ax = plt.subplots(
    nrows=3,
    ncols=4,
    sharex=True,
    sharey=True, )

ax = ax.flatten()

for i in range(12):
    # 只查看了前面12张图片
    img = mnist_train.data[i]
    ax[i].imshow(img)
    ax[i].set(title=get_fashion_mnist_labels(mnist_train[i][1]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

batch_size = 256

def get_dataloader_workers():
    """使用四个进程读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers())
def load_data_fashion_mnist(batch_size,resize=None):
    """下载Fashion-MNIST数据集，并将其保存至内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize)) # transforms.Resize将图片最小的一条边缩放到指定大小，另一边缩放对应比例
    trans = transforms.Compose(trans) # compose用于串联多个操作
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                           num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test,batch_size,shuffle=True,
                          num_workers = get_dataloader_workers()))
