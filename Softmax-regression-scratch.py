# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:55:24 2024

@author: 11693
"""

import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
test_iter.num_workers = 0
train_iter.num_workers = 0
num_inputs = 784  
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义Softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp / partition  # 利用了广播机制进行矩阵运算

# 实现Softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)
# w.shape[0]表示w的0维（行）大小

# 先来介绍怎么在所有的预测值中根据标号把对应的预测值找出来
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5]])
y_hat[[0, 1], y]
"""[0，1]指的是真实样本的下标，对于第0个样本，拿出y[0]样本类别的预测值(y[0] = 0,对应0.1)，
对于第1个样本，拿出y[1]样本类别的预测值(y[1] = 2,对应0.5)。拿出真实标号类的预测值。"""

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # y_hat这里用来从'y_hat'张量中选择与真实标签'y'相对应的预测概率

cross_entropy(y_hat, y)

# 比较预测值和真实y，计算出分类正确的类别数，最终计算准确率
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 元素最大的那个下标存到y_hat里面，这里的每一行其实对应的就是每一个样本
        y_hat = y_hat.argmax(axis = 1)
    #把y_hat转为y的数据类型再与y做比较，存入cmp
    cmp = y_hat.type(y.dtype) == y
    #返回预测正确的aggravate
    return float(cmp.type(y.dtype).sum())  # 再将布尔类型的cmp转为y的类型，求和
accuracy(y_hat,y)/len(y)  # y的长度其实就是总样本数


def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度，用于初步评估"""
    if isinstance(net, torch.nn.Module):
        """这个if判断是为了确保只有继承自torch.nn.Module的模型才会被切换到评估模式。在PyTorch中，继承自
        torch.nn.Module的模型可以包含许多不同类型的层和操作，其中一些层在训练和评估阶段的行为是不同的。"""
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 这个函数目的就是累加，每次累加到0和1的位置上，0就是正确预测数，1就是预测数量。
    with torch.no_grad():
        """确保在评估模型精度时不会计算参数的梯度，以提高计算效率。这在只需要模型的前向传播输出的情况下是一种很常见的做法。"""
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]        

evaluate_accuracy(net, test_iter)

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  # updater需要是sgd优化器
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 如果是自定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
lr = 0.1

def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)