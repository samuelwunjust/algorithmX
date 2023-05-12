import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Dropout, Flatten
from torch.utils.data import DataLoader
import algorithmX
from torchvision import datasets, transforms



def load_data():

    train_data = datasets.FashionMNIST("../datasets", train=True,
                                       transform=transforms.Compose(
                                           [transforms.Resize(size=224), transforms.ToTensor()]),
                                       download=True)
    test_data = datasets.FashionMNIST("../datasets", train=False,
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

    return train_data, test_data, train_data_size, test_data_size, train_dataloader, test_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the test and train load
train_data, test_data, train_data_size, test_data_size, train_dataloader, test_dataloader = load_data()
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)


# define the alexnet

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
    print("the total loss on the test data{:.3f}". format(total_test_loss))
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


net = nn.Sequential(
  
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
   
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
   
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 10))

train_to_device(net, train_dataloader, test_dataloader, 10, 0.001, device)



