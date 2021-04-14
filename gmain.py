import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# 读取数据的函数,先读取标签，再读取图片
def _read(image, label):
    """解压标签包"""
    with gzip.open(label) as flbl:
        ''' 采用Big Endian的方式读取两个int类型的数据，且参考MNIST官方格式介绍，magic即为magic number (MSB first) 
        用于表示文件格式，num即为文件夹内包含的数据的数量'''
        magic, num = struct.unpack(">II", flbl.read(8))
        '''将标签包中的每一个二进制数据转化成其对应的十进制数据，且转换后的数据格式为int8（-128 to 127）格式，返回一个数组'''
        label = np.frombuffer(flbl.read(), dtype=np.int8)
    '''以只读形式解压图像包'''
    with gzip.open(image, 'rb') as fimg:
        '''采用Big Endian的方式读取四个int类型数据，且参考MNIST官方格式介绍，magic和num上同，rows和cols即表示图片的行数和列数'''
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        '''将图片包中的二进制数据读取后转换成无符号的int8格式的数组，并且以标签总个数，行数，列数重塑成一个新的多维数组'''
        image = np.frombuffer(fimg.read(), dtype=np.uint8)
        image = image.reshape(len(label), rows, cols)
    return image, label


# 读取数据
def get_data():
    train_img, train_label = _read(
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')

    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [train_img, train_label, test_img, test_label]


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.layer2conv = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.layer3conv = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.layer4line = nn.Linear(in_features=120, out_features=84)
        ''' 添加Dropout层，阈值暂设置为0.5 '''
        self.layer5dropout = nn.Dropout(0.5)
        ''' 问题是二分类问题，因此将out_features值从10改为2 '''
        self.layer5line = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.layer1conv(x))  # 6*28*28
        x = self.maxpool(x)  # 6*14*14
        x = F.relu(self.layer2conv(x))  # 16*10*10
        x = self.maxpool(x)  # 16*5*5
        x = F.relu(self.layer3conv(x))  # 120*1*1
        x = x.view(x.size(0), -1)  # 120
        x = F.relu(self.layer4line(x))  # 84
        x = self.layer5line(x)  # 2
        return x


network = Network()
print(network)
''' 训练结果可视化 '''
x = []
y1 = []
''' 加载已经训练好的模型并在现在的网络上运行 '''
if os.path.exists('network_data.pth'):
    network.load_state_dict(torch.load('network_data.pth'))
''' 获取数据并解包 '''
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = get_data()
''' 将解包来的训练数据封装成float32的多维矩阵 '''
train_set_x_orig = torch.tensor(train_set_x_orig, dtype=torch.float32) / 128 - 1
test_set_x_orig = torch.tensor(test_set_x_orig, dtype=torch.float32) / 128 - 1
''' 将解包来的训练数据对应结果封装成int64的多维矩阵 '''
train_set_y_orig = torch.tensor(train_set_y_orig, dtype=torch.int64)
test_set_y_orig = torch.tensor(test_set_y_orig, dtype=torch.int64)
''' 对测试集进行维度扩充 '''
test_x = test_set_x_orig.unsqueeze_(1)
pred_test = network(test_x)
''' 表示第二列(第1维度)要消失，为了使保留最大值位置信息 '''
maxcorrect = pred_test.argmax(dim=1).eq(test_set_y_orig).sum().item()
''' 使用Adam优化算法，利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率，并规定学习率为0.0001 '''
optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.0001)
''' batch_size = 600 '''
for t in range(550):
    train_x = train_set_x_orig[100 * t:100 * t + 100].unsqueeze_(1)
    train_y = train_set_y_orig[100 * t:100 * t + 100]
    pred_train = network(train_x)
    ''' 计算交叉熵损失，描述概率分布的差异信息 '''
    train_loss = F.cross_entropy(pred_train, train_y)
    ''' 优化器的梯度置0 '''
    optimizer.zero_grad()
    ''' 反向传播运算 '''
    train_loss.backward()
    ''' 更新参数空间 '''
    optimizer.step()
    print(train_loss.item())
    pred_test = network(test_x)
    correct = pred_test.argmax(dim=1).eq(test_set_y_orig).sum().item()
    ''' 进行可视化统计 '''
    x.append(t)
    y1.append(train_loss.item())
    ''' 如训练得更好的模型则会保存 '''
    if correct > maxcorrect:
        torch.save(network.state_dict(), 'network_data.pth')
        maxcorrect = correct
''' 显示图像 '''
plt.xlim(0, 550)
plt.ylim(0, 2.5)
plt.plot(np.array(x), np.array(y1), "r", label="Modified")
plt.show()