import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module
import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Dropout, Flatten
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda")


class accmulat_contain:
    def __init__(self, n):
        self.datas = [0.0] * n

    def __add__(self, *adds):
        self.datas = [x + float(y) for x, y in zip(self.datas, adds)]

    def clear(self):
        self.datas = [0.0] * len(self.datas)

    def get(self, idx):
        return self.datas[idx]


def load_data():
    # 准备数据集
    train_data = datasets.FashionMNIST("../data", train=True,
                                       transform=transforms.Compose(
                                           [transforms.Resize(size=224), transforms.ToTensor()]),
                                       download=True)
    test_data = datasets.FashionMNIST("../data", train=False,
                                      transform=transforms.Compose(
                                          [transforms.Resize(size=224), transforms.ToTensor()]),
                                      download=True)
    #  trainsform the shape of image from28*28 -> 224*224

    # return the size of train_data_size
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    # load the data and define the batch_size of data
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    return train_dataloader, test_dataloader, train_data_size, test_data_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the test and train load
train_dataloader, test_dataloader, train_data_size, test_data_size = load_data()


def evaluate_accuracy(net, test_dataloader):
    net.eval()
    loss = nn.CrossEntropyLoss()

    loss = loss.to(device)
    test_step = 0
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            XX, YY = data
            XX = XX.to(device)
            YY = YY.to(device)
            YY_hat = net(XX)
            l = loss(YY_hat, YY)
            total_test_loss += l
            total_test_accuracy += (YY_hat.argmax(1) == YY).sum()
            test_step += 1
    print("the total loss on the test data{:.3f}".format(total_test_loss))
    print("the average accuracy on the test data{:.3f}".format(total_test_accuracy / test_data_size))


def train_to_device(net, train_dataloader, test_dataloader, epochss, lr, device):
    net.to(device)

    loss = nn.CrossEntropyLoss()  # 定义loss
    loss = loss.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 优化器

    train_step = 0

    for epochs in range(epochss):
        net.train()

        print('the {}th train '.format(epochs + 1))
        for data in train_dataloader:
            X, Y = data
            Y = Y.to(device)
            X = X.to(device)
            Y_hat = net(X)  # create hat

            l = loss(Y_hat, Y)  # compute the loss
            optimizer.zero_grad()  # 1
            l.backward()  # 2
            optimizer.step()  # 3
            train_step += 1
            if train_step % 100 == 0:
                print("the train times{:.3f},loss value{:.3f}".format(train_step, l))

        evaluate_accuracy(net, test_dataloader)


def vgg_block(num_convs, in_channels, out_channels):  # define the vgg blocks
    blocks = []

    for i in range(num_convs):
        blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blocks.append(nn.ReLU())
        in_channels = out_channels
    blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blocks)


def vgg(num_arg):
    conlk = []
    in_channels = 1
    # make the conv part
    for (num_convs, out_channels) in num_arg:
        conlk.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    # make the linear part
    return nn.Sequential(
        *conlk, nn.Flatten(),

        nn.Linear(out_channels * 7 * 7, 4096),  # 每个块使得高宽减半，通道数翻倍

        nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10)

    )


def AlexNet():
    return nn.Sequential(

        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))


class Inception(nn.Module):

    def __init__(self,in_channels,c1,c2,c3,c4,**krwarg):

        super(Inception,self).__init__(**krwarg)
        self.path1_1=nn.Conv2d(in_channels,c1,kernel_size=1)
        self.path2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.path2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        self.path3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.path3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        self.path4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.path4_2=nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.path1_1(x))
        p2 = F.relu(self.path2_2(F.relu(self.path2_1(x))))
        p3=F.relu(self.path3_2(F.relu(self.path3_1(x))))
        p4=F.relu(self.path4_2(self.path4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)


def GoogleLetNet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())
    return  nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


