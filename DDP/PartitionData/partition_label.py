# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月18日 1:14
"""
import math
from random import Random

import h5py
import torch
import torch.distributed as dist
import numpy as np
from DDP.MakeDataset.h5_to_dataset import train_dataset
from DDP.configs import *


class Partition_by_label(object):
    def __init__(self, data, label_list):
        self.data = data
        self.label_list = label_list  # 当前节点所拥有的目标

        # 从总的数据集中抽出本节点的数据
        data_temp = []
        label_temp = []
        for da, lab in self.data:
            if lab.item() in self.label_list:
                data_temp.append(np.array(da))
                label_temp.append(lab.item())
        # print(len(data_temp))
        data_temp = np.array(data_temp)
        label_temp = np.array(label_temp)
        self.data_temp = data_temp
        self.data = torch.utils.data.TensorDataset(torch.tensor(data_temp), torch.tensor(label_temp))

    def __len__(self):
        return len(self.data_temp)  # 返回一个节点所拥有的数据条数

    def __getitem__(self, index):
        return self.data[index]


class DataPartition_by_label(object):
    def __init__(self, data, total_target=8, seed=1234):
        self.data = data
        self.total_target = total_target
        self.partitions = []

        size = dist.get_world_size()  # 获取节点总数
        np.random.seed(seed)
        label_len = math.ceil(self.total_target / size)
        indexes = [x for x in range(0, total_target)]
        np.random.shuffle(indexes)

        for frac in range(size):
            if (frac != size - 1):
                self.partitions.append(indexes[0:label_len])
                indexes = indexes[label_len:]
            else:
                temp = indexes[0:]
                # temp.append(self.partitions[0][0:2])
                self.partitions.append(temp)

        # print(self.partitions)
        # print(len(self.partitions))

    def use(self, partition):
        print(f"target list{self.partitions[partition]}, of Rank{partition}.")
        return Partition_by_label(self.data, self.partitions[partition])


def partition_dataset_by_label(dataset, batch_num, num_target):
    """ Partitioning dataset """
    size = dist.get_world_size()  # 获取节点（GPU）数量
    partition = DataPartition_by_label(dataset, total_target=num_target)
    # print(dist.get_rank())
    partition = partition.use(dist.get_rank())  # 获得本进程分得的部分数据
    len_data = partition.__len__()

    # 获取batch_size使得每个节点上的batch数量一样
    if len_data % batch_num == 0:
        bsz = len_data / batch_num
    else:
        bsz = len_data / (batch_num-1)

    output_set = torch.utils.data.DataLoader(partition,
                                             batch_size=int(bsz),
                                             shuffle=True)

    return output_set, int(bsz), len_data


if __name__ == '__main__':
    # from DDP.MakeDataset.h5_to_dataset import train_dataset
    #
    data_path = '/home/wha/DML_code/matlab_code/train_8target_10mode_21-Apr-2022.h5'
    train_data, val_data = train_dataset(data_path)
    partition_dataset_by_label(train_data, 512, 8)

