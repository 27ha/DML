from __future__ import print_function
import argparse
from itertools import zip_longest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import numpy as np
import wandb


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_epoch_sgd(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test_epoch_sgd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    wandb.log({'test_loss': test_loss, 'accuracy': 100. * correct / len(test_loader.dataset)})


def train_epoch_dsgd(args, models, device, train_loaders, optimizers, schedulers, epoch):
    n_agents = len(models)
    for model in models:
        model.train()

    for batch_idx, data_targets in enumerate(zip_longest(*train_loaders)):
        old_models = deepcopy(models)
        for i in range(n_agents):
            if data_targets[i] is not None:
                data, target = data_targets[i]
                data, target = data.to(device), target.to(device)
                optimizers[i].zero_grad()
                output = models[i](data)
                loss = F.nll_loss(output, target)
                loss.backward()
                _temp_models = [None] * n_agents
                for k in range(n_agents):
                    if k != i:
                        _temp_models[k] = models[k]
                        models[k] = old_models[k]
                with torch.no_grad():
                    for ps in zip(*[models[j].parameters() for j in range(n_agents)]):
                        new_val = torch.mean(torch.stack(ps), dim=0)
                        new_val -= schedulers[i].get_lr()[0] * ps[i].grad
                        ps[i].copy_(new_val)
                for k in range(n_agents):
                    if k != i:
                        models[k] = _temp_models[k]
                if batch_idx % args.log_interval == 0:
                    print('Agent: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, epoch, batch_idx *
                            len(data), len(train_loaders[i].dataset),
                        100. * batch_idx / len(train_loaders[i]), loss.item()))
                    if args.dry_run:
                        break


def test_epoch_dsgd(models, device, test_loaders):
    for model in models:
        model.eval()
    n_agents = len(models)
    test_losses = [0] * n_agents
    correctes = [0] * n_agents
    with torch.no_grad():
        for i in range(n_agents):
            for data, target in test_loaders[i]:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # sum up batch loss
                test_losses[i] += F.nll_loss(output,
                                             target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correctes[i] += pred.eq(target.view_as(pred)).sum().item()

    n_tests = np.sum([len(test_loaders[i].dataset) for i in range(n_agents)])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        np.sum(test_losses)/n_tests, np.sum(correctes), n_tests, 100. * np.sum(correctes) / n_tests))
    
    wandb.log({'test_loss': np.sum(test_losses)/n_tests, 'accuracy': 100. * np.sum(correctes) / n_tests})


def train_sgd(transform, args, train_kwargs, test_kwargs, device):
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_epoch_sgd(args, model, device, train_loader, optimizer, epoch)
        test_epoch_sgd(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def train_dsgd(transform, args, train_kwargs, test_kwargs, device):
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=transform)

    labels = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    n_agents = len(labels)
    train_indices = []
    test_indices = []
    for i in range(n_agents):
        train_ind = None
        test_ind = None
        for l in labels[i]:
            if train_ind is None:
                train_ind = train_dataset.targets == l
                test_ind = test_dataset.targets == l
            train_ind |= train_dataset.targets == l
            test_ind |= test_dataset.targets == l
        train_indices.append(train_ind)
        test_indices.append(test_ind)
    sub_train_datasets = [deepcopy(train_dataset) for i in range(n_agents)]
    sub_test_datasets = [deepcopy(test_dataset) for i in range(n_agents)]
    for train_ind, test_ind, sub_train_dataset, sub_test_dataset in zip(
            train_indices, test_indices, sub_train_datasets, sub_test_datasets):
        sub_train_dataset.data, sub_train_dataset.targets = sub_train_dataset.data[
            train_ind], sub_train_dataset.targets[train_ind]
        sub_test_dataset.data, sub_test_dataset.targets = sub_test_dataset.data[
            test_ind], sub_test_dataset.targets[test_ind]

    train_loaders = [torch.utils.data.DataLoader(
        sub_train_dataset, **train_kwargs) for sub_train_dataset in sub_train_datasets]
    test_loaders = [torch.utils.data.DataLoader(
        sub_test_dataset, **train_kwargs) for sub_test_dataset in sub_test_datasets]

    models = [Net().to(device) for i in range(n_agents)]
    optimizers = [optim.Adadelta(
        model.parameters(), lr=args.lr/10) for model in models]
    schedulers = [StepLR(optimizer, step_size=1, gamma=args.gamma)
                         for optimizer in optimizers]
    for epoch in range(1, args.epochs + 1):
        train_epoch_dsgd(args, models, device, train_loaders,
                         optimizers, schedulers, epoch)
        test_epoch_dsgd(models, device, test_loaders)
        for i in range(n_agents):
            schedulers[i].step()

    if args.save_model:
        for i in range(n_agents):
            torch.save(models[i].state_dict(), "mnist_cnn_{i}.pt".format(i))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--alg', type=str, default='dsgd',
                        help='the optimization method, selects from [sgd, dsgd] (default: dsgd)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.init(project="l2do-cnn-mnist", group=args.alg)

    torch.manual_seed(args.seed)

    device=torch.device("cuda" if use_cuda else "cpu")

    train_kwargs={'batch_size': args.batch_size}
    test_kwargs={'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs={'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.alg == 'sgd':
        train_sgd(transform, args, train_kwargs, test_kwargs, device)
    elif args.alg == 'dsgd':
        train_dsgd(transform, args, train_kwargs, test_kwargs, device)


if __name__ == '__main__':
    main()
