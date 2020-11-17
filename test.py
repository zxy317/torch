import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models import get_model

# 参考train.py的注释
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
# 参考train.py的注释
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = get_model(name_model="lenet", num_classes=10)  # 创建网络实例,分类类别有10类
PATH = './checkpoint/cifar_net_best.pth' # 训练的模型存储路径
net.load_state_dict(torch.load(PATH)) # 获取模型参数

correct = 0 # 测试集总正确分类数量
total = 0 # 测试集数据总量
class_correct = list(0. for i in range(10)) # 每个类别测试集正确分类数量
class_total = list(0. for i in range(10)) # 每个类别测试集总数据量
net.eval() # 固定模型，测试集不改变权值
for data in testloader:
    images, labels = data # 获取数据和标签
    outputs = net(images) # 模型分类结果
    _, predicted = torch.max(outputs.data, 1) # torch.max函数获得概率最大的分类结果，加上_,才能将predicted转化成类别序号
    total += labels.size(0) # labels.size(0)为4，即要给mini-batch的数量
    correct += (predicted == labels).sum().item() # 获取一个mini-batch中正确分类的结果数量
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

# 打印总准确率
print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
# 打印各个类别准确率
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))