import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

class LeNet_car(nn.Module):

    def __init__(self):
        super(LeNet_car, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.conv3 = nn.Conv2d(24, 36, 5)

        self.fc1 = nn.Linear(24 * 4 * 4 , 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128, 6)
        self.pool = nn.MaxPool2d(3, 3)

    # forward这个函数定义了前向传播的运算
    # 流程：卷积->relu->压缩->全连接层->relu->最后再一次全连接层
    def forward(self, x):
        # 第一层卷积并做非线性变换
        x = F.relu(self.conv1(x))
        # 结果进行压缩
        x = self.pool(x)
        # 第二层卷积并做非线性变换
        x = F.relu(self.conv2(x))
        # 再压缩
        x = self.pool(x)

        # 把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 24 * 4 * 4)

        # 先做线性变换后做非线性变换
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def car_classification_modelload():
    net_model_car = torch.load('./car_classify/data/net_model_car.pt')
    net_model_car.eval()
    return net_model_car


