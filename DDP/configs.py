# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年03月18日 16:43

网络及训练参数的配置信息
"""
# --------------------------------- 用于训练的GPU的数量，并会在每个GPU上创建一个进程 ---------------------------------
RANKS = 3

# --------------------------------- 网络结构的参数配置 ---------------------------------
INPUT_CHANNEL = 2
CLASS_NUM = 8  # 类别数
EPOCHS = 100
BATCH_SIZE = 128 * RANKS
BATCH_NUM = 108     # 用于数据按标签分时，统一每个节点的batch数量(108是保证每个节点的batch_size不大于128，需要根据数据分布来设置)
LEARN_RATE = 0.01

# --------------------------------- 划分数据方式 ---------------------------------
BY_RANDOM = True
BY_TARGET = False

# --------------------------------- 保存文件名 ---------------------------------
SAVE_LOSS = "loss_radio_424_randomSample"
SAVE_ACC = "acc_radio_424_randomSample"
SAVE_CONFUSION_MATRIX = "conf_radio_424_randomSample"
SAVE_RECORD = "record_radio_424_randomSample"
SAVE_MODEL = "model_radio_424_randomSample"

# SAVE_LOSS = "loss_radio_424_partLabel"
# SAVE_ACC = "acc_radio_424_partLabel"
# SAVE_CONFUSION_MATRIX = "conf_radio_424_partLabel"
# SAVE_RECORD = "record_radio_424_partLabel"
# SAVE_MODEL = "model_radio_424_partLabel"

# --------------------------------- 保存路径 ---------------------------------
TRAIN_LOSS_PATH = "/home/wha/DML_code/DDP/Project/Loss"
TRAIN_ACC_PATH = "/home/wha/DML_code/DDP/Project/Accuracy"
TRAIN_CONFUSION_MATRIX_PATH = "/home/wha/DML_code/DDP/Project/Confusion_Matrix"
TRAIN_RECORD_PATH = "/home/wha/DML_code/DDP/Project/Record"
TRAIN_MODEL_PATH = "/home/wha/DML_code/DDP/Project/Model"

# --------------------------------- 数据集路径 ---------------------------------
TRAIN_PATH = '/home/wha/DML_code/matlab_code/train_8target_10mode_21-Apr-2022.h5'
TEST_PATH = '/home/wha/matlab_code/train_8target_10mode_21-Apr-2022.h5'

# --------------------------------- 设置早停点 ---------------------------------
EARLY_STOP = 50

# --------------------------------- 记录loss，记录accuracy ---------------------------------
loss_on_train_list = []
acc_on_train_list = []
acc_on_val_list = []
loss_on_val_list = []





TEMP_TEST = "model_mobile"

# SampleSize = [3, 32, 32]       # 样本的尺寸[通道数，长，宽]
