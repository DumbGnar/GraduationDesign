import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
from torch.utils.data import DataLoader
import models
import settings as st
import matplotlib.pyplot as plt
import numpy as np

all_image_list = []
class_id = []

''' 设置训练集总长度 '''
# data_length = st.rows
data_length = 3499
data_label = [-1] * data_length
''' 进行地址寻访 '''
prev_dir = st.base + "data_pictures\\training\\"
after_dir = '.jpg'

''' 进行数据读入，id为文件夹名称 '''
for id in range(2):
    id_string = str(id)
    for filename in glob(prev_dir + id_string +'\\*.jpg'):
        print(filename)
        ''' 提取出图片编号，并设置data_label[编号] = value '''
        position = filename.replace(prev_dir+id_string+'\\', '')
        position = position.replace(after_dir, '')
        print(position)
        data_label[int(position)] = id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use device ', device)

''' 生成网络 '''
net = models.Network()
net = net.float()
net = net.to(device=device)

''' 使用交叉熵损失函数 '''
criterion = nn.CrossEntropyLoss()
''' 更新使用算法为随机梯度下降，并设置学习率添加L2正则 '''
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

''' 设置文件路径 '''
imgs_dir = prev_dir
''' 读入数据 '''
all_data = models.Reader(imgs_dir, data_label)
train_loader = DataLoader(all_data, batch_size=st.BATCH_SIZE, shuffle=True)
print('finish loading data, ready to load')

''' 模型训练 '''
x = []
y = []
counts = 0
for epoch in range(10):
    net.train()

    epoch_loss = 0.0
    batch_num = 0
    for training_batch in train_loader:
        batch_num = batch_num + 1

        images = training_batch['image']
        labels = training_batch['label']

        ''' 迁移至GPU '''
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        ''' 梯度归零初始化 '''
        optimizer.zero_grad()

        outputs = net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        ''' 每20条数据打印一次 '''
        if batch_num % 20 == 19:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_num + 1, epoch_loss / 200))
            ''' 记录下图像点 '''
            x.append(counts)
            counts += 1
            y.append(epoch_loss / 200)
            epoch_loss = 0.0

''' 打印图像 '''
plt.plot(np.array(x), np.array(y), "r", label="Modified")
plt.show()
