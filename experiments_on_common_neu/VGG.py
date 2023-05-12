from torch import nn, optim
import sys
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Dropout, Flatten
from torch.utils.data import DataLoader
import algorithmX
from torchvision import datasets, transforms
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):  # define the vgg blocks
    blocks = []

    for i in range(num_convs):
        blocks.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        blocks.append(nn.ReLU())
        in_channels=out_channels
    blocks.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blocks)


def vgg(num_arg):
    conlk=[]
    in_channels=1
    #make the conv part
    for (num_convs,out_channels) in num_arg:
        conlk.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels
    #make the linear part
    return nn.Sequential(
        *conlk,nn.Flatten(),

        nn.Linear(out_channels*7*7,4096),#每个块使得高宽减半，通道数翻倍

        nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,10)


    )
X = torch.randn(size=(1, 1, 224, 224))
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

net=vgg(conv_arch)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
device=("cuda")
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
algorithmX.train_to_device(net, train_iter, test_iter, num_epochs, lr, device)
