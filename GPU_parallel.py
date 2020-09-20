import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters 和 DataLoaders
input_size = 5 # 一次输入有5个数据
output_size = 2 # 对应数据为2个数据

batch_size = 30 # 一个batch有30组数据
data_size = 100 # 一共有100组数据

# 定义第一个设备为可见cuda设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 自定义加载数据集
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# 制作一个虚拟(随机）数据集
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

# CNN模型结构
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size) # 建立模型实例
print(torch.cuda.device_count()) # 打印可用GPU数量
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model) # 用多个GPU来加速训练

model.to(device) # 将模型传递到GPU

# 训练过程
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())