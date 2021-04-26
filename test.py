import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
import models
import settings as st

all_image_list = []
class_id = []

''' 设置训练集总长度 '''
# data_length = st.rows
''' 读取临时保存的字典 '''
temp_dict = np.load("temp.npy", allow_pickle=True)
data_length = temp_dict.item()["test_counts"]
data_label = [-1] * data_length
''' 进行地址寻访 '''
prev_dir = st.base + "data_pictures\\testing\\"
after_dir = '.jpg'

''' 进行数据读入，id为文件夹名称 '''
for id in range(2):
    id_string = str(id)
    for filename in glob(prev_dir + id_string + '\\*.jpg'):
        print(filename)
        ''' 提取出图片编号，并设置data_label[编号] = value '''
        position = filename.replace(prev_dir + id_string + '\\', '')
        position = position.replace(after_dir, '')
        print(position)
        data_label[int(position)] = id

# if there is GPU, choose GPU else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use device ', device)

''' 生成网络 '''
net = models.Network()
net = net.float()
net.to(device=device)

all_data = models.Reader(prev_dir, data_label)

batch_size = 4

test_loader = DataLoader(all_data, batch_size=batch_size, shuffle=True)

print('Load data finish, ready to train')

PATH = './model.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data['image']
        labels = data['label']

        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {0} test images: {1}'.format(total, 100 * correct / total))