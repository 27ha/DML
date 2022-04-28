# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月18日 10:08
"""

import torch
import torch.utils.data as data
import numpy as np
from random import Random


def shuffle_data(original_data, min_num):
    """
    打乱原始数据，取前min_num条
    :param original_data: 原始数据
    :param min_num: 抽取数量
    :return: 抽取后得到的数据
    """
    length_data = len(original_data)
    index_list = []
    for index in range(length_data):
        index_list.append(index)
    rng = Random()
    rng.seed(1357)
    rng.shuffle(index_list)
    out_data = []
    for n in range(min_num):
        out_data.append(original_data[index_list[n]])
    return out_data


# 文件存放路径
path = '/home/wha/DML_code/DDP/data/test/'

# 文件名称列表
dataName_list = ['honor10_1.npy', 'honor10_2.npy', 'honor10_4.npy', 'huaweip9_1.npy', 'huaweip9_4.npy',
                 'iphone6s_1.npy', 'iphone6s_2.npy', 'iphone6s_5.npy', 'meizux8_1.npy', 'oppor11_2.npy',
                 'oppor11_4.npy', 'xaiomi6_3.npy', 'xiaomi6_1.npy']

# 存放数据
data_list = []
# 存放标签
label_list = []

# 查看最小数据的条数, iphone6s_1.npy: 8061 * 8192
dataPath = path + dataName_list[10]
tempData = np.load(dataPath)
min_length = len(tempData)

temp_label = 0
for dataName in dataName_list:
    data_path = path + dataName
    temp_data = shuffle_data(np.load(data_path), min_length)

    # 将每条样本 1*8192 变成 2*64*64
    for i in range(min_length):
        data_list.append(temp_data[i].reshape(2, 64, 64))
        label_list.append(temp_label)
    temp_label += 1

test_data = data.TensorDataset(torch.tensor(np.array(data_list)), torch.tensor(np.array(label_list)))
