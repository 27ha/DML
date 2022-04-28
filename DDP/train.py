# -*- coding: UTF-8 -*-
"""
@author： ha
@create： 2022年04月10日 20:31
"""
# 导入系统库
from datetime import datetime
from math import ceil
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist

# 导入自己写的库
from DDP.configs import *
from DDP.MakeDataset import train_dataset
from DDP.MakeDataset import h5_to_dataset
import resnet
import DDP.PartitionData.partition_batch as pb
import DDP.PartitionData.partition_label as pl
from PlotFigure.plot_loss_acc import *

'''
torch.distributed 软件包为在一台或多台计算机上运行的多个计算节点之间的多进程并行性提供了 PyTorch 支持和通信原语。
torch.nn.parallel.DistributedDataParallel() 类基于此功能构建，以提供同步分布式训练作为任何 PyTorch 模型的包装器。
'''


def init_process(rank, size, epochs, fn, backend='gloo'):
    '''
    :param group:
    :param rank: 进程号，即第几个进程
    :param size: 总的进程数
    :param epochs: 训练轮数
    :param fn: 训练函数，即run函数
    :param backend: 通信后端，默认gloo
    :return:

    --环境变量初始化--
    MASTER_PORT - required; has to be a free port on machine with rank 0
    MASTER_ADDR - required (except for rank 0); address of rank 0 node
    WORLD_SIZE - required; can be set either here, or in a call to init function
    RANK - required; can be set either here, or in a call to init function
    '''

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # MASTER_PORT：计算机上的可用端口，它将承载等级为 0 的进程。
    # MASTER_ADDR：将承载等级为 0 的进程的计算机的 IP 地址。
    # 确保每个进程都能够通过主站使用相同的IP地址和端口进行协调

    '''
    在调用任何其他方法之前，需要使用 torch.distributed.init_process_group() 函数初始化包。这将阻塞，直到所有进程都已加入。
    # WORLD_SIZE：进程的总数，以便主节点知道要等待多少个工作线程。
    # RANK：每个进程的排名，以便他们知道它是否是工人的主人。
    '''
    dist.init_process_group(backend, world_size=size, rank=rank)

    fn(rank, size, epochs)


# --------------------------------- 创建所需文件夹 ---------------------------------
os.makedirs(TRAIN_RECORD_PATH, exist_ok=True)
os.makedirs(TRAIN_MODEL_PATH, exist_ok=True)
os.makedirs(TRAIN_ACC_PATH, exist_ok=True)
os.makedirs(TRAIN_LOSS_PATH, exist_ok=True)
os.makedirs(TRAIN_CONFUSION_MATRIX_PATH, exist_ok=True)

# --------------------------------- 加载训练集及验证集 ---------------------------------
print("*"*20, "start loading data!!!", "*"*20)
train_data, val_data = h5_to_dataset.train_dataset(TRAIN_PATH)
# train_data = train_dataset.train_data
# val_data = train_dataset.val_data


def dataNormalization(input):
    """ normalization of data"""
    input = abs(input)
    input_max = torch.max(input)
    input_min = torch.min(input)
    # input = abs(input)
    Max_Min = input_max - input_min
    output = torch.div(input - input_min, Max_Min)
    return output


def cal_acc(output, target):
    """ calculating accuracy """
    acc = 0.0
    out1 = output.cpu().detach().numpy()
    for j in range(len(target)):
        acc = acc + (np.argmax(out1[j]) == target[j].cpu().numpy())
    return acc / len(target)


def val_loss_acc(in_data, model_, cuda_num):
    loss_ = 0.0
    acc_ = 0.0
    n = 0  # 记录in_data中的batch数量
    with torch.no_grad():
        for da, lab in in_data:
            n += 1
            da = dataNormalization(da)
            da, lab = Variable(da.cuda(cuda_num).float()), Variable(lab.cuda(cuda_num).float())
            out_ = model_(da)
            acc_ += cal_acc(output=out_, target=lab)
            loss_ += F.nll_loss(out_, lab.long()).item()
        acc_ = acc_ / n
        loss_ = loss_ / n
    return loss_, acc_


def average_param(model):
    """ averaging gradient """
    size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def average_acc_loss(acc, loss, acc_val, loss_val, cuda_num):
    ''' averaging acc and loss that from different node'''
    temp_acc = torch.tensor(acc).cuda(cuda_num).float()
    temp_loss = torch.tensor(loss).cuda(cuda_num).float()
    temp_acc_val = torch.tensor(acc_val).cuda(cuda_num).float()
    temp_loss_val = torch.tensor(loss_val).cuda(cuda_num).float()

    dist.all_reduce(temp_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(temp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(temp_acc_val, op=dist.ReduceOp.SUM)
    dist.all_reduce(temp_loss_val, op=dist.ReduceOp.SUM)

    mean_acc = temp_acc / dist.get_world_size()
    mean_loss = temp_loss / dist.get_world_size()
    mean_acc_val = temp_acc_val / dist.get_world_size()
    mean_loss_val = temp_loss_val / dist.get_world_size()

    acc_on_train_list.append(mean_acc.item())
    loss_on_train_list.append(mean_loss.item())
    acc_on_val_list.append(mean_acc_val.item())
    loss_on_val_list.append(mean_loss_val.item())

    return mean_acc


def run(rank, size, epochs):
    """ Distributed Synchronous SGD Example """
    # """Blocking point-to-point communication."""
    # tensor = torch.zeros(1)
    # req = None
    # if rank == 0:
    #     tensor += 1
    #     # Send the tensor to process 1
    #     req = dist.send(tensor=tensor, dst=1)
    # elif rank == 1:
    #     # Receive tensor from process 0
    #     req = dist.recv(tensor=tensor, src=0)
    #     tensor += 1
    #     # Send the tensor to process 2
    #     req = dist.send(tensor, dst=2)
    # else:
    #     # Receive tensor from process 0
    #     req = dist.recv(tensor, src=1)
    # print(tensor[0], "in rank", rank)

    # """Non-blocking point-to-point communication."""
    # tensor = torch.zeros(1)
    # req = None
    # if rank == 0:
    #     tensor += 1
    #     req = dist.isend(tensor=tensor, dst=1)
    #     print('Rank 0 started sending')
    # elif rank == 1:
    #     req = dist.irecv(tensor=tensor, src=0)
    #     print('Rank 1 started receiving')
    #     tensor += 1
    #     req = dist.isend(tensor=tensor, dst=2)
    #     print('Rank 1 started sending')
    # else:
    #     req = dist.irecv(tensor=tensor, src=1)
    #     print('Rank 2 started receiving')
    # req.wait()
    # print(tensor[0], " in Rank", rank, end='\n')

    # 初始化进程
    # """ All-Reduce example."""
    # """ Simple collective communication. """
    # group = dist.new_group(ranks=[0, 1, 2])
    # tensor = torch.ones(1)
    # dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=group)
    # # print(tensor[0], " in Rank", rank, end="")
    # print(f"{tensor[0]} in Rank{rank}")
    max_acc = 0.0

    # --------------------------------- 指定训练的gpu编号 ---------------------------------
    cuda_num = rank + 0

    torch.manual_seed(1007)

    # --------------------------------- 按划分方式获取本进程的数据集 ---------------------------------
    print("")
    if BY_RANDOM:
        train_set, train_bsz, len_train_set = pb.partition_dataset(train_data, batch_size=BATCH_SIZE)
        val_set, val_bsz, len_val_set = pb.partition_dataset(val_data, batch_size=BATCH_SIZE)
    if BY_TARGET:
        train_set, train_bsz, len_train_set = pl.partition_dataset_by_label(train_data, batch_num=BATCH_NUM, num_target=CLASS_NUM)
        val_set, val_bsz, len_val_set = pl.partition_dataset_by_label(val_data, batch_num=BATCH_NUM, num_target=CLASS_NUM)

    # --------------------------------- 加载模型 -----------------------------------
    model = resnet.ResNet_101()
    model = model.cuda(cuda_num)

    # --------------------------------- 定义优化器 ---------------------------------
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARN_RATE, momentum=0.5,
                          weight_decay=1e-2)    # 使用l2正则化

    num_batches = ceil(len_train_set / float(train_bsz))  # 获取batch的数量
    print("Rank", dist.get_rank(), " start to train!!!")
    count = 0  # 用于统计模型连续未变优的次数，用于早停
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch, (data, target) in enumerate(train_set):
            start = datetime.now()
            data = dataNormalization(data)
            data, target = Variable(data.cuda(cuda_num).float()), Variable(target.cuda(cuda_num).float())
            optimizer.zero_grad()
            output = model(data)
            acc = cal_acc(output=output, target=target)
            epoch_acc += acc
            loss = F.nll_loss(output, target.long())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # --------------------------------- AllReduce模型参数 ---------------------------------
            average_param(model)

            # --------------------------------- 计算全部验证集的loss及acc ---------------------------------
            loss_val, acc_val = val_loss_acc(in_data=val_set, model_=model, cuda_num=cuda_num)

            end = datetime.now()

            # --------------------------------- 打印信息 ---------------------------------
            print(f'|| epoch: {epoch:0>{len(str(epochs))}}/{epochs} -- '
                  f'batch: {batch+1:0>{len(str(num_batches))}}/{num_batches} -- '
                  f'train accuracy: {acc * 100:08.4f}% -- '
                  f'train loss: {loss.item():.10f} -- '
                  f'validate accuracy: {acc_val * 100:08.4f}% -- '
                  f'validate loss: {loss_val:.10f} -- '
                  f'spend time: {(end - start).seconds:.4f}, "s" -- '
                  f'in Rank {dist.get_rank()} ||  ')

            # --------------------------------- 将打印信息写入txt文件 ---------------------------------
            f = open('/home/wha/DML_code/DDP/Project/Record/' + SAVE_RECORD + '.txt', 'a', encoding="utf-8")
            print(f'|| epoch: {epoch:0>{len(str(epochs))}}/{epochs} -- '
                  f'batch: {batch+1:0>{len(str(num_batches))}}/{num_batches} -- '
                  f'train accuracy: {acc * 100:08.4f}% -- '
                  f'train loss: {loss.item():.10f} -- '
                  f'validate accuracy: {acc_val * 100:08.4f}% -- '
                  f'validate loss: {loss_val:.10f} -- '
                  f'spend time: {(end - start).seconds:.4f}, "s" -- '
                  f'in Rank {dist.get_rank()} ||  ', flush=True, file=f)

        # --------------------------------- 计算训练集loss及acc ---------------------------------
        epoch_acc = epoch_acc / num_batches
        epoch_loss = epoch_loss / num_batches

        # --------------------------------- 在节点0上平均所有节点的loss及acc, 并保存到相应的list中 ---------------------------------
        mean_acc = average_acc_loss(epoch_acc, epoch_loss, acc_val, loss_val, cuda_num)

        # --------------------------------- 在节点0上保存最优模型 ---------------------------------
        if rank == 1:
            if mean_acc > max_acc:
                count = 0
                max_acc = mean_acc
                torch.save(model.state_dict(),
                           TRAIN_MODEL_PATH + '/' + SAVE_MODEL + '.pkl')
            else:
                count += 1

        # --------------------------------- 在节点0上画出loss图和acc图 ---------------------------------
        if rank == 1:
            plot_loss(loss_list=loss_on_train_list, val_loss_list=loss_on_val_list, is_train=True)
            plot_acc(acc_list=acc_on_train_list, val_acc_list=acc_on_val_list, is_train=True)

        # --------------------------------- 早停 ---------------------------------
        if count >= EARLY_STOP:
            print('The model has converged.')
            break


def main():
    # --------------------------------- 用于训练的GPU的数量，并会在每个GPU上创建一个进程 ---------------------------------
    size_of_rank = RANKS

    processes = []  # 存放进程
    mp.set_start_method("spawn")
    for rank in range(size_of_rank):
        p = mp.Process(target=init_process, args=(rank, size_of_rank, EPOCHS, run, 'nccl'))
        # target表示调用对象，即子进程要执行的任务
        # args表示调用对象的位置参数元组，args = (1, 2, 'egon',)
        p.start()
        processes.append(p)

    '''
    p.start()：启动进程，并调用该子进程中的p.run()
    p.run()：进程启动时运行的方法，正是它去调用target指定的函数，我们自定义类的类中一定要实现该方法
    p.terminate()：强制终止进程p，不会进行任何清理操作，如果p创建了子进程，该子进程就成了僵尸进程，使用该方法需要特别小心这种情况。如果p还保存了一个锁那么也将不会被释放，进而导致死锁
    p.is_alive()：如果p仍然运行，返回True
    p.join(\\\[timeout\\\])：主线程等待p终止（强调：是主线程处于等的状态，而p是处于运行的状态）。
                            timeout是可选的超时时间，需要强调的是，p.join只能join住start开启的进程，而不能join住run开启的进程
    '''
    for p in processes:
        p.join()



if __name__ == '__main__':
    main()
