# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月18日 11:06
"""

# 混淆矩阵
import itertools
from decimal import Decimal
import numpy as np
from matplotlib import pyplot as plt

from DDP.configs import *


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# 绘制混淆矩阵
def plot_confusion_matrix(epoch, totle_num, cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    font_size = 20
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm1 = cm.astype('float') / totle_num
        # thresh = cm.max() / 2./totle_num
    plt.figure(figsize=(28, 20))
    plt.imshow(cm1, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=font_size + 5)
    cb = plt.colorbar()
    # cb.set_label('colorbar', fontsize=font_size)
    cb.ax.tick_params(labelsize=font_size)  # 设置色标刻度字体大小。
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=90)
    plt.tick_params(labelsize=font_size)
    plt.yticks(tick_marks, classes)

    plt.xticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        # num = '{:.2f}'.format((int(cm[i, j]))/totle_num) if normalize else int(cm[i, j])
        num = '{:.2f}'.format(Decimal((int(cm[i, j])) / totle_num)) if normalize else int(cm[i, j])

        # plt.text(j, i, num,
        #          verticalalignment='center',
        #          horizontalalignment="center",
        #          color="white" if num > thresh else "black")
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="red" if float(num) > float(0) else "white",
                 fontsize=font_size
                 # color="red"
                 )

    plt.tight_layout()
    plt.ylabel('Predicted label', fontsize=font_size + 5)
    plt.xlabel('True label', fontsize=font_size + 5)
    plt.savefig(TRAIN_CONFUSION_MATRIX_PATH + '/' + SAVE_CONFUSION_MATRIX + str(epoch) + '.png', bbox_inches='tight')
    plt.close()