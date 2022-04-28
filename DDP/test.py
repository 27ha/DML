# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月18日 10:01
"""

import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import resnet
# from DDP.MakeDataset.test_dataset import test_data
from DDP.MakeDataset import h5_to_dataset
from DDP.configs import *


def MaxMinNormalization(input_data):
    input_data = abs(input_data)
    input_max = torch.max(input_data)
    input_min = torch.min(input_data)
    Max_Min = input_max - input_min
    output = torch.div(input_data-input_min, Max_Min)
    return output


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# --------------------------------- 加载模型 ---------------------------------
model = resnet.ResNet_18().to(device)
model.load_state_dict(torch.load(TRAIN_MODEL_PATH + '/' + SAVE_MODEL + '.pkl'))

# --------------------------------- 加载数据 ---------------------------------
test_data = h5_to_dataset.test_dataset(TEST_PATH)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# 存放 loss
test_loss = []
# 存放 accuracy
test_acc = []

for i, (da, lab) in enumerate(test_loader):
    with torch.no_grad():
        da = MaxMinNormalization(da)
        da = Variable(da)
        lab = Variable(lab)
        da = da.to(torch.float32)
        da = da.to(device)
        lab = lab.to(device).to(torch.float32)
        out = model(da)

        # --------------------------------- Loss ---------------------------------
        loss_temp = F.nll_loss(out, lab.long())  # [24, 1]
        test_loss.append(loss_temp.cpu())

        # --------------------------------- Accuracy ---------------------------------
        va_rightCount = 0
        out1 = out.cpu().detach().numpy()
        for j in range(len(lab)):
            va_rightCount = va_rightCount + int(np.argmax(out1[j]) == lab[j].cpu().numpy())
        va_acc = va_rightCount / len(lab)
        test_acc.append(va_acc)

plt.figure(1)
plt.plot(np.array(test_loss), 'coral')
plt.xlabel('the result of Test')
plt.legend(['Loss'])
plt.show()

plt.figure(2)
plt.plot(np.array(test_acc), 'yellowgreen')
plt.xlabel('the result of Test')
plt.legend(['Accuracy'])
plt.show()