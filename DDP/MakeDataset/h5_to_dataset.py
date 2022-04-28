# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月21日 16:00
"""

import h5py
import torch
import torch.utils.data as data
import numpy as np


def train_dataset(train_path):
    f = h5py.File(train_path, 'r', libver='latest', swmr=True)

    train_da_list = []
    train_lab_list = []
    valid_da_list = []
    valid_lab_list = []

    for key in f.keys():
        # print('key:', key)
        # print('the length of key{}: {}'.format(key, len(f[key][:, 0, 0, 0])))
        # print('*' * 50)
        data_temp_list = []
        label_temp_list = []
        for j in range(len(f[key][:, 0, 0, 0])):
            data_temp = f[key][j, :, :, :]
            label_temp = int(key) - 1
            data_temp_list.append(data_temp)
            label_temp_list.append(label_temp)

        length = len(data_temp_list)

        # shuffle the list of data
        np.random.seed(1007)
        np.random.shuffle(data_temp_list)

        # 80% of the data is train data
        train_tdata_list = data_temp_list[:int(0.8 * length)]
        train_tlabel_list = label_temp_list[:int(0.8 * length)]
        train_da_list.extend(train_tdata_list)
        train_lab_list.extend(train_tlabel_list)

        # 20% of the data is valid data
        valid_tda_list = data_temp_list[int(0.8 * length):]
        valid_tlab_list = label_temp_list[int(0.8 * length):]
        valid_da_list.extend(valid_tda_list)
        valid_lab_list.extend(valid_tlab_list)

    train_data = data.TensorDataset(torch.tensor(np.array(train_da_list)), torch.tensor(np.array(train_lab_list)))
    valid_data = data.TensorDataset(torch.tensor(np.array(valid_da_list)), torch.tensor(np.array(valid_lab_list)))
    return train_data, valid_data


def test_dataset(test_path):
    f = h5py.File(test_path, 'r')

    data_list = []
    label_list = []

    for key in f.keys():
        # print('key:', key)
        # print('the length of key{}: {}'.format(key, len(f[key][:, 0, 0, 0])))
        # print('*' * 50)
        for j in range(len(f[key][:, 0, 0, 0])):
            data_temp = f[key][j, :, :, :]
            label_temp = int(key) - 1
            data_list.append(data_temp)
            label_list.append(label_temp)

    test_data = data.TensorDataset(torch.tensor(data_list), torch.tensor(label_list))
    return test_data


if __name__ == '__main__':
    data_path = '/home/wha/DML_code/matlab_code/train_8target_10mode_21-Apr-2022.h5'
    train_data, val_data = train_dataset(data_path)
    print(train_data[0])