import h5py
import torch
import torch.utils.data as data
import random
import tqdm
import numpy as np

path = '/home/wha/DML_code/matlab_code/train_8target_10mode_21-Apr-2022.h5'
f = h5py.File(path, 'r')

data_list = []
label_list = []
train_da_list = []
train_lab_list = []
valid_da_list = []
valid_lab_list = []
test_da_list = []
test_lab_list = []

for key in f.keys():
    # print('-'*50)
    print('key:', key)
    print('the length of key{}: {}'.format(key, len(f[key][:, 0, 0, 0])))
    #print(f[key][:])
    #data_list[key] = (f[key][:], key)
    #print((torch.tensor(f[key][:]), int(key)))
    data_temp_list = []
    label_temp_list = []
    for j in range(len(f[key][:, 0, 0, 0])):
        #print(j)
        #print(len(f[key][:, 0, 0, 0]))
        #print(f[key][j, :, :, :])
        data_temp = f[key][j, :, :, :]
        label_temp = int(key) - 1
        #print(temp)
        data_temp_list.append(data_temp)
        label_temp_list.append(label_temp)

    data_list.extend(data_temp_list)
    label_list.extend(label_temp_list)
    length = len(data_temp_list)

    # shuffle the list of data
    random.shuffle(data_temp_list)

    # 60% of the data is train data
    train_tdata_list = data_temp_list[:int(0.6*length)]
    train_tlabel_list = label_temp_list[:int(0.6*length)]
    train_da_list.extend(train_tdata_list)
    train_lab_list.extend(train_tlabel_list)

    # 20% of the data is valid data
    valid_tda_list = data_temp_list[int(0.6*length):int(0.8*length)]
    valid_tlab_list = label_temp_list[int(0.6*length):int(0.8*length)]
    valid_da_list.extend(valid_tda_list)
    valid_lab_list.extend(valid_tlab_list)

    # 20% of the data is test data
    test_tda_list = data_temp_list[int(0.8*length):]
    test_tlab_list = label_temp_list[int(0.8*length):]
    test_da_list.extend(test_tda_list)
    test_lab_list.extend(test_tlab_list)



'''data_list = torch.tensor(data_list)
label_list = torch.tensor(label_list)'''
print(len(data_list))
train_Data = data.TensorDataset(torch.tensor(data_list), torch.tensor(label_list))

train_data = data.TensorDataset(torch.tensor(train_da_list), torch.tensor(train_lab_list))
valid_data = data.TensorDataset(torch.tensor(valid_da_list), torch.tensor(valid_lab_list))
test_data = data.TensorDataset(torch.tensor(test_da_list), torch.tensor(test_lab_list))

if __name__ == '__main__':
    train_Data = data.DataLoader(train_Data, batch_size=360, shuffle=True)
    for i, (x, y) in enumerate(train_Data):
        print('-'*20)
        print(i)
        print(x.size())
        print(y.size())
    '''print(len(train_Data))
    for i, x in enumerate(train_Data):
        print('i:', i)
        print(x)'''

    '''for i, (x, y) in enumerate(data_list):
        print(x)
        print(y)
        break'''

    '''for i, (x, y) in enumerate(valid_data):
        print(x)
        print(y)
        print(x.size())
        print('-'*20)'''