import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from models import get_model

# class Net(nn.Module):
#     # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
#     def __init__(self):
#         super(Net, self).__init__()  # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
#         self.conv1 = nn.Conv2d(3, 6, 5)  # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
#         self.pool1 = nn.MaxPool2d(2, 2)  # 定义池化函数，在2*2的观察区域内取最大值作为采样数据进行降维操作，这样做的优点是可以使显性特征更明显，减少计算开销。
#         self.conv2 = nn.Conv2d(6, 16, 5)  # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
#         self.pool2 = nn.MaxPool2d(2, 2)  # 定义池化函数，在2*2的观察区域内取最大值作为采样数据进行降维操作，这样做的优点是可以使显性特征更明显，减少计算开销。
#         # self.fc1 = nn.Linear(16 * 5 * 5, 10)  # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到10个节点上。
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
#         self.fc2 = nn.Linear(120, 84)  # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
#         self.fc3 = nn.Linear(84, 10)  # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
#
#     # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pool1(x)  # 输入x经过卷积conv1之后，经过激活函数ReLU（原来这个词是激活函数的意思），使用2x2的窗口进行最大池化Max pooling，然后更新到x。
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool2(x)  # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
#         x = x.view(-1, 16 * 5 * 5)  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
#         # x = self.fc1(x)  # 输入x经过全连接3，然后更新x
#         x = F.relu(self.fc1(x))  # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
#         x = F.relu(self.fc2(x))  # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
#         x = self.fc3(x)  # 输入x经过全连接3，然后更新x
#         return x

def train(epoch):

    start = time.time()
    net.train()
    running_loss = 0.
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，参考https://www.runoob.com/python/python-func-enumerate.html
    for i, data in enumerate(trainloader, 0):
        # 获得输入数据，inputs和labels分别都是四组，为一个mini-batch
        inputs, labels = data

        # 清零梯度缓存
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # 网络输出结果
        loss = criterion(outputs, labels)  # 根据labels计算loss值
        loss.backward()  # 反向传播误差
        optimizer.step()  # 更新参数，默认SGD（随机梯度下降）策略

        # print statistics
        running_loss += loss.item()  # loss是标量，item()就是调用该标量的值
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 2000))
            running_loss = 0.0

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    correct = 0.0  # 测试集总正确分类数量
    total = 0  # 测试集数据总量

    net.eval()  # 固定模型，测试集不改变权值
    for data in testloader:
        images, labels = data  # 获取数据和标签
        outputs = net(images)  # 模型分类结果
        _, predicted = torch.max(outputs.data, 1)  # torch.max函数获得概率最大的分类结果，加上_,才能将predicted转化成类别序号
        total += labels.size(0)  # labels.size(0)为4，即要给mini-batch的数量
        correct += (predicted == labels).sum().item()  # 获取一个mini-batch中正确分类的结果数量

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        correct / len(testloader.dataset),
        finish - start
    ))
    print()
    return correct / len(testloader.dataset)

if __name__ == '__main__':




    net = get_model(name_model="vgg", num_classes=10)  # 创建网络实例,分类类别有10类
    criterion = nn.CrossEntropyLoss()  # 多分类的交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降优化器(使用momentum）

    # torchvision输出的是PILImage，值的范围是[0,1]。
    # 我们将其转化为tensor数据，并归一化为[-1,1]。
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
    # 1.root，表示cifar10数据的加载的相对目录
    # 2.train，表示是否加载数据库的训练集，false的时候加载测试集
    # 3.download，表示是否自动下载cifar数据集
    # 4.transform，表示是否需要对数据进行预处理，none为不进行预处理
    # 数据集已经提前下载到本地./data路径下，所以不需要去官网下载，只需要把D:\anaconda3\Lib\site-packages\torchvision\datasets\cifar.py的路径改成本地./data路径即可
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 将训练集的图片划分成许多份mini-batch，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=0表示使用主进程来加载数据
    # 如果num_workers设置为2，会报错RuntimeError，原因是多进程需要在main函数中运行，解决方法：1.加main函数，在main中调用；2.num_workers改为0，单进程加载
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    # test数据集的加载过程与train数据集同理
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    # CIFAR10数据集十个类别
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0] and best_acc < acc:
            PATH = './checkpoint/cifar_net_best.pth'
            torch.save(net.state_dict(), PATH)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            PATH = './checkpoint/cifar_net_regular.pth'
            torch.save(net.state_dict(), PATH)


    print('Finished Training')

    # 保存已训练得到的模型
    PATH = './cifar_net_final.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Saving')
