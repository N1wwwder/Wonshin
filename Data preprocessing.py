# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 01:00:16 2024

@author: 11693
"""

import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建一个新目录,生成一个相对路径指向data
data_file = os.path.join('..', 'data', 'house_tiny.csv')  # 设置文件路径指向‘house_tiny.csv'
with open(data_file, 'w') as f:  # 打开文件用于写入
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    
import pandas as pd
data = pd.read_csv(data_file)  # 使用 pandas 的 read_csv 函数读取 CSV 文件中的数据。
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only = True))
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y


