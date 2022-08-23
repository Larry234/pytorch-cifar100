""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(net, num_class=100, device=None):
    """ return given network
    """

    if net == 'vgg16':
        from models.vgg import vgg16_bn
        model = vgg16_bn(num_class=num_class)
    elif net == 'vgg13':
        from models.vgg import vgg13_bn
        model = vgg13_bn(num_class=num_class)
    elif net == 'vgg11':
        from models.vgg import vgg11_bn
        model = vgg11_bn(num_class=num_class)
    elif net == 'vgg11cp':
        from models.vgg import vgg11_CP
        model = vgg11_CP(num_class=num_class)
    elif net == 'vgg11cpb':
        from models.vgg import vgg11_CPB
        model = vgg11_CPB(num_class=num_class)
    elif net == 'vgg11em':
        from models.vgg import vgg11_EM
        model = vgg11_EM(num_class=num_class)
    elif net == 'vgg11fcp':
        from models.vgg import vgg11_FCP
        model = vgg11_FCP(num_class=num_class)
    elif net == 'vgg16cp':
        from models.vgg import vgg16_bn_CP
        model = vgg16_bn_CP(num_class=num_class)
    elif net == 'vgg16cpb':
        from models.vgg import vgg16_bn_CPB
        model = vgg16_bn_CPB(num_class=num_class)
    elif net == 'vggsmall':
        from models.custom import vgg_small
        model = vgg_small()
    elif net == 'vggsmallcp':
        from models.custom import vgg_small_CP
        model = vgg_small_CP()
    elif net == 'vggsmallcpb':
        from models.custom import vgg_small_CPB
        model = vgg_small_CPB()
    elif net == 'vggmedium':
        from models.custom import vgg_medium
        net = vgg_medium()
    elif net == 'vggmediumcp':
        from models.custom import vgg_medium_CP
        model = vgg_medium_CP()
    elif net == 'vggmediumcpb':
        from models.custom import vgg_medium_CPB
        model = vgg_medium_CPB()
    elif net == 'vgg19':
        from models.vgg import vgg19_bn
        model = vgg19_bn()
    elif net == 'densenet121':
        from models.densenet import densenet121
        model = densenet121()
    elif net == 'densenet161':
        from models.densenet import densenet161
        model = densenet161()
    elif net == 'densenet169':
        from models.densenet import densenet169
        model = densenet169()
    elif net == 'densenet201':
        from models.densenet import densenet201
        model = densenet201()
    elif net == 'googlenet':
        from models.googlenet import googlenet
        model = googlenet()
    elif net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        model = inceptionv3()
    elif net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        model = inceptionv4()
    elif net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        model = inception_resnet_v2()
    elif net == 'xception':
        from models.xception import xception
        model = xception()
    elif net == 'resnet18':
        from models.resnet import resnet18
        model = resnet18()
    elif net == 'resnet34':
        from models.resnet import resnet34
        model = resnet34()
    elif net == 'resnet50':
        from models.resnet import resnet50
        model = resnet50()
    elif net == 'resnet101':
        from models.resnet import resnet101
        model = resnet101()
    elif net == 'resnet152':
        from models.resnet import resnet152
        model = resnet152()
    elif net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        model = preactresnet18()
    elif net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        model = preactresnet34()
    elif net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        model = preactresnet50()
    elif net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        model = preactresnet101()
    elif net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        model = preactresnet152()
    elif net == 'resnext50':
        from models.resnext import resnext50
        model = resnext50()
    elif net == 'resnext101':
        from models.resnext import resnext101
        model = resnext101()
    elif net == 'resnext152':
        from models.resnext import resnext152
        model = resnext152()
    elif net == 'shufflenet':
        from models.shufflenet import shufflenet
        model = shufflenet()
    elif net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        model = shufflenetv2()
    elif net == 'squeezenet':
        from models.squeezenet import squeezenet
        model = squeezenet()
    elif net == 'mobilenet':
        from models.mobilenet import mobilenet
        model = mobilenet()
    elif net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        model = mobilenetv2()
    elif net == 'nasnet':
        from models.nasnet import nasnet
        model = nasnet()
    elif net == 'attention56':
        from models.attention import attention56
        model = attention56()
    elif net == 'attention92':
        from models.attention import attention92
        model = attention92()
    elif net == 'seresnet18':
        from models.senet import seresnet18
        model = seresnet18()
    elif net == 'seresnet34':
        from models.senet import seresnet34
        model = seresnet34()
    elif net == 'seresnet50':
        from models.senet import seresnet50
        model = seresnet50()
    elif net == 'seresnet101':
        from models.senet import seresnet101
        model = seresnet101()
    elif net == 'seresnet152':
        from models.senet import seresnet152
        model = seresnet152()
    elif net == 'wideresnet':
        from models.wideresidual import wideresnet
        model = wideresnet()
    elif net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        model = stochastic_depth_resnet18()
    elif net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        model = stochastic_depth_resnet34()
    elif net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        model = stochastic_depth_resnet50()
    elif net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        model = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    return model


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]