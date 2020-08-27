import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np

class MapNet(nn.Module):

    def __init__(self):
        super(MapNet, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        nn.init.normal_(self.fc3.weight, std=0.001)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mlp_model(guard):
    model=torch.load('cfg/mlp_' + guard + '.pt')
    return model

def position_prediction(net_model,box,guard='left'):
    """
    :param net_model: pretrained model
    :param guard: left or right
    :param box: (xyxy) 图像中某个car的box坐标
    :return: （x,y） MLP映射的实际地图坐标
    """

    with torch.no_grad():
        img_w = (box[2] - box[0])
        img_h = (box[3] - box[1])
        inputs = [box[0] + img_w / 2, box[3], img_w, img_h]
        inputs=torch.tensor(inputs,dtype=torch.float32)

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # 计算前向传播的输出
        outputs = net_model(inputs)
        mlp_out = outputs.cpu().numpy()
        x = np.clip(mlp_out[0], 0, 8)
        y = np.clip(mlp_out[1], 0, 5)
    return x,y

def position_fusion(pos_data,x_shift=0,y_shift=0):
    """
    如果有解算位置，则和mlp位置进行融合
    :param pos_data:
    :param x_shift: mlp位置输出可能与真值有固定的偏移
    :return:
    """
    if len(pos_data['position']) != 0 and np.max(pos_data['position']) != 0 and len(pos_data['armor_box'])!=0 :
        car_x1, car_y1, car_x2, car_y2 = pos_data['car_box'].astype('int')
        arm_x1, arm_y1, arm_x2, arm_y2 = pos_data['armor_box'][0]
        # 确保装甲板在车内
        if car_x1 < arm_x1 < car_x2 and car_y1 < arm_y1 < arm_y2:
            # 取出第一个解算位置
            for i in range(len(pos_data['position'])):
                if np.max(pos_data['position'][i]) != 0:
                    cal_out = pos_data['position'][i]
                    break
            mlp_out = pos_data['position_mlp']
            #mlp_out[0] = mlp_out[0] + x_shift
            #mlp_out[1] = mlp_out[1] + y_shift
            # 限制mlp输出范围
            mlp_out = np.array(mlp_out)
            mlp_out[0] = np.clip(mlp_out[0], 0, 8)
            mlp_out[1] = np.clip(mlp_out[1], 0, 5)

            fusion_result = mlp_out[:]
            if mlp_out[0] < 0.5 or mlp_out[0] == 8 or mlp_out[1] < 0.5 or mlp_out[1] == 5:
                fusion_result = cal_out

            return fusion_result
        else:
            return None
    else:
        return None

def compute_l2(output,label):
    """
    :param output: (n,2)
    :param label: (n,2)
    :return:
    """
    dis=output-label
    error=np.sqrt(dis[:,0]**2+dis[:,1]**2)
    return sum(error)/len(label)

