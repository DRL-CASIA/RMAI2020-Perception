import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms




# 定义网络===============================================================================================================
class LeNet_angle(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet_angle, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.conv3 = nn.Conv2d(24, 36, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所                                                                                                                  以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(24 * 4 * 4 + 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # 最终有4类，所以最后一个全连接层输出数量是4
        self.fc4 = nn.Linear(128, 8)
        self.pool = nn.MaxPool2d(3, 3)

    # forward这个函数定义了前向传播的运算
    # 流程：卷积->relu->压缩->全连接层->relu->最后再一次全连接层
    def forward(self, x, y):
        # 第一层卷积并做非线性变换
        x = F.relu(self.conv1(x))
        # 结果进行压缩
        x = self.pool(x)
        # 第二层卷积并做非线性变换
        x = F.relu(self.conv2(x))
        # 再压缩
        x = self.pool(x)
        # 第三层卷积并做非线性变换
        # x = F.relu(self.conv3(x))
        # # 再压缩
        # x = self.pool(x)
        # print(x.shape)
        # 把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 24 * 4 * 4)
        y1 = torch.tensor(y, dtype=torch.float32).cuda()

        cat = torch.cat((x, y1), 1)
        # print(x.shape)
        # 先做线性变换后做非线性变换
        cat = F.relu(self.fc1(cat))
        cat = F.relu(self.fc2(cat))
        cat = F.relu(self.fc3(cat))
        cat = self.fc4(cat)
        return cat
