# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月18日 11:18
"""

import torch.distributed as dist
from matplotlib import pyplot as plt
from DDP.configs import *


def plot_acc(acc_list=[36.2843, 61.7606, 89.0270], val_acc_list=[41.2109, 64.6804, 91.1335], is_train=False):
    # 准确率acc图
    plt.figure(figsize=(10, 5))
    if is_train:
        plt.title("Accuracy During Training")
    plt.plot(acc_list, label="train_acc", color='coral')
    plt.plot(val_acc_list, label="val_acc", color='yellowgreen')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(TRAIN_ACC_PATH + '/' + SAVE_ACC + '.jpg')
    plt.close()


def plot_loss(loss_list, val_loss_list, is_train=False):
    # 损失loss图
    plt.figure(figsize=(10, 5))
    if is_train:
        plt.title("Loss During Training")
    plt.plot(loss_list, label="train_loss", color='coral')
    plt.plot(val_loss_list, label="val_loss", color='yellowgreen')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(TRAIN_LOSS_PATH + '/' + SAVE_LOSS + '.jpg')
    plt.close()


if __name__ == '__main__':
    plot_acc(is_train=True)