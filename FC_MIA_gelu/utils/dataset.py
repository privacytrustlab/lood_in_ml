import torch
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

import numpy as np

def mnist(batch_size, data_augmentation = True, shuffle = True, valid_ratio = None):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST('./data/MNIST', train = True, download = True, transform = transform)
    validset = datasets.MNIST('./data/MNIST', train = True, download = True, transform = transform)
    testset = datasets.MNIST('./data/MNIST', train = False, download = True, transform = transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if valid_ratio is not None and valid_ratio > 0.:

        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)

        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, valid_loader, test_loader, classes

    else:
        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(100 * 10)
        train_idx = indices[:split_pt]
        train_sampler = SubsetRandomSampler(train_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, None, test_loader, classes

def cifar10(batch_size, data_augmentation = False, shuffle = False, valid_ratio = None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]) if data_augmentation == True else transforms.Compose([transforms.ToTensor()])
    transform_valid = transforms.Compose([
        transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset = datasets.CIFAR10(root = './data/cifar10', train = True, download = True, transform = transform_train)
    validset = datasets.CIFAR10(root = './data/cifar10', train = True, download = True, transform = transform_valid)
    testset = datasets.CIFAR10(root = './data/cifar10', train = False, download = True, transform = transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if valid_ratio is not None and valid_ratio > 0.:

        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)

        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        assert shuffle == True, 'shuffle must be True with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 4, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 4, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

        return train_loader, valid_loader, test_loader, classes

    else:
        trainset.targets = torch.LongTensor(trainset.targets)
        testset.targets = torch.LongTensor(testset.targets)
        
        
        idx = trainset.targets < 2
        trainset.targets = trainset.targets[idx]
        trainset.data = trainset.data[idx]
        idx = testset.targets < 2
        testset.targets = testset.targets[idx]
        testset.data = testset.data[idx]

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = 4, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)


        return train_loader, None, test_loader, classes

def FashionMNIST(batch_size, data_augmentation = True, shuffle = True, valid_ratio = None):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.FashionMNIST('./data/FashionMNIST', train = True, download = True, transform = transform)
    validset = datasets.FashionMNIST('./data/FashionMNIST', train = True, download = True, transform = transform)
    testset = datasets.FashionMNIST('./data/FashionMNIST', train = False, download = True, transform = transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if valid_ratio is not None and valid_ratio > 0.:

        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)

        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, valid_loader, test_loader, classes

    else:
        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(100 * 10)
        train_idx = indices[:split_pt]
        train_sampler = SubsetRandomSampler(train_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'
        #
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, None, test_loader, classes