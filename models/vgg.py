"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

from .ConcatPool import *


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], # 0.6814
    'A1': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'C'], # 0.6770
    'A2': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,          'CB'], # 0.6762
    'A3': [64,    'KM', 128,     'KM', 256, 256,          'KM', 512, 512,          'KM', 1024, 1024,        'KM'],
    'AE': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'E'], # 0.6775
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B1': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'C'],
    'B2': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,          'CB'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'C'],
    'D2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,     'CB'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100, init_weights=True, CP=False):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if CP:
            self.classifier = nn.Sequential(
                nn.Linear(2048 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_class)
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        output = self.features(x)
        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        # output = output.view(x.size(0), -1)
        output = self.classifier(output)

        return output
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        elif l == 'C':
            layers += [ConcatPooling2d(kernel_size=2, stride=2)]
            continue
        elif l == 'CB':
            layers += [ConcatPooling2d(kernel_size=2, stride=2), nn.Conv2d(2048, 512, kernel_size=1, stride=1)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def make_layersFCP(cfg, batch_norm=True):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        elif l == 'KM':
            layers += [KMaxPooling2d(kernel_size=2, stride=2)]
            continue
            
        if input_channel == 3:
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            input_channel = l
        else:
            layers += [nn.Conv2d(l, l, kernel_isze=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

    

def vgg11_bn(num_class=100):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_class=num_class)

def vgg11_CP(num_class=100):
    return VGG(make_layers(cfg['A1'], batch_norm=True), num_class=num_class, CP=True)

def vgg11_CPB(num_class=100):
    return VGG(make_layers(cfg['A2'], batch_norm=True), num_class=num_class)

def vgg11_FCP(num_class=100):
    return VGG(make_layersFCP(cfg['A3'], batch_norm=True), CP=True)

def vgg11_EM(num_class=100):
    return VGG(make_layers(cfg['AE'], batch_norm=True), num_class=num_class)

def vgg13_bn(num_class=100):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_class=num_class)

def vgg16_bn(num_class=100):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_class)

def vgg16_bn_CP(num_class=100):
    return VGG(make_layers(cfg['D1'], batch_norm=True), num_class=num_class, CP=True)

def vgg16_bn_CPB(num_class=100):
    return VGG(make_layers(cfg['D2'], batch_norm=True), num_class=num_class)

def vgg19_bn(num_class=100):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_class=num_class)


