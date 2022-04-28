# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月18日 1:14
"""
from random import Random
import torch
import torch.distributed as dist

""" Dataset partitioning helper """


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartition(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1007):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        # print(len(self.partitions))

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, batch_size):
    """ Partitioning dataset """
    size = dist.get_world_size()  # 获取节点（GPU）数量
    bsz = batch_size / float(size)  # 均分batch_size的数据, 即每个节点batch的大小
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartition(dataset, partition_sizes)
    # print(dist.get_rank())
    partition = partition.use(dist.get_rank())  # 获得本进程分得的部分数据
    output_set = torch.utils.data.DataLoader(partition,
                                             batch_size=int(bsz),
                                             shuffle=True)
    return output_set, int(bsz), partition.__len__()
