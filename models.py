import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Reader(Dataset):
    def __init__(self, img_path, data_label):
        """ 设置图像地址以及标签 """
        self.imgs_dir = img_path
        self.ids = data_label

    def __len__(self):
        """ 返回标签长度 """
        return len(self.ids)

    def __getitem__(self, index):
        """ 返回第index位的data_label数据 """
        ''' 获取该index对应的结果值 '''
        res = self.ids[index]
        file_path = self.imgs_dir + str(res) + "\\" + str(index) + ".jpg"
        ''' 读入图片并调整 '''
        img = Image.open(file_path).convert('L')
        img = np.array(img)
        img = img.reshape(1, 32, 32)
        ''' 使用cross entropy优化算法要对亮度值做归一化处理 '''
        if img.max() > 1:
            img = img / 255
        return {"image": torch.from_numpy(img), "label": torch.tensor(res)}


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
