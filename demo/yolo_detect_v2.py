# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:39:18 2020

@author: Administrator
"""
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
# from models_nolambda import *
from models_nolambda_focallossw import *
from utils.parse_config import *
from preprocess import prep_image, inp_to_image, letterbox_image
from utils.utils_mulanchor import *
import pandas as pd
import random 
import pickle as pkl
import argparse
from PIL import Image


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim[1], input_dim[0]))  # resize: w h
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]  # w h
    img = (letterbox_image(orig_im, (inp_dim[1], inp_dim[0])))  # orig_im 352 608
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, color_dict):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    #color = random.choice(colors)
    color = color_dict[str(cls)]
    
    if cls <= 22:
        cv2.rectangle(img, c1, c2,color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


def output(img, CUDA, model, device,num_classes,confidence=0.05, nms_thesh=0.02,inp_dim=[416,416]):
    img = np.array(img)
    img, orig_im, dim = prep_image(img, inp_dim)  # resize img
    im_dim = torch.FloatTensor(dim).repeat(1, 2)
    # tensor([[512., 256., 512., 256.]])

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    with torch.no_grad():
        output = model(Variable(img)).to(device)

    output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

    im_dim = im_dim.repeat(output.size(0), 1)
    # tensor([[512., 256., 512., 256.],
    #         [512., 256., 512., 256.],
    #         [512., 256., 512., 256.]], device='cuda:0')

    scaling_factor_w = torch.min(inp_dim[1] / im_dim, 1)[0].view(-1, 1)

    scaling_factor_h = torch.min(inp_dim[0] / im_dim, 1)[0].view(-1, 1)


    output[:, [1, 3]] -= (inp_dim[1] - scaling_factor_w * im_dim[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim[0] - scaling_factor_w * im_dim[:, 1].view(-1, 1))/2

    output[:, [1, 3]] /= scaling_factor_w
    output[:, [2, 4]] /= scaling_factor_w

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    output_dicts = []
    for i in range(output.shape[0]):
        if output[i, -1] == 0.0:
            output_dict ={'car_box': np.array(output[i, 1: 5].detach().cpu()), 'armor_box': np.array([])}
            output_dicts.append(output_dict)
    for i in range(output.shape[0]):
        if output[i, -1] != 0.0:
            for j in range(len(output_dicts)):
                box1 = np.array(output[i, 1: 5].detach().cpu())
                box2 = output_dicts[j]['car_box']
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
                inter_rect_x1 = max(b1_x1, b2_x1)
                inter_rect_y1 = max(b1_y1, b2_y1)
                inter_rect_x2 = min(b1_x2, b2_x2)
                inter_rect_y2 = min(b1_y2, b2_y2)
                # Intersection area
                inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, abs(inter_rect_x2 - inter_rect_x1 + 1)) * np.clip(
                    inter_rect_y2 - inter_rect_y1 + 1, 0, abs(inter_rect_y2 - inter_rect_y1 + 1)
                )  # clamp: 将input中的元素限制在[min,max]范围内并返回一个Tensor  torch.clamp(input,min,max,out=None)
                # Union Area
                b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
                b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

                iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
                if iou > 0.01:
                    output_dicts[j]['armor_box'] = np.append(output_dicts[j]['armor_box'], np.array(output[i, 1: 5].detach().cpu())).reshape((-1, 4))

    return output_dicts
    # return np.array(output[:, 1: 5].detach().cpu()), np.array(output[:, -1].detach().cpu())


'''
# 注：
# 输出为一个list,list中包含每辆车的字典,目前字典中有两个key值,'car_box'与'armor_box', car_box为一维数组, armor_box为二维数组.‘armor_box’为[]时没有匹配到对应的装甲板.匹配的iou阈值可以调整.
# 格式如：[{'car_box': array([468.017  ,  86.88042, 526.57666, 138.35327], dtype=float32), 'armor_box': array([], dtype=float64)}, {'car_box': array([382.3557 , 167.36795, 459.72476, 228.34549], dtype=float32), 'armor_box': array([[394.31442261, 204.36643982, 415.21707153, 218.80717468],
#        [442.17236328, 205.49127197, 459.47769165, 221.09608459]])}, {'car_box': array([ 63.237453, 135.55783 , 137.73201 , 192.92749 ], dtype=float32), 'armor_box': array([[112.04547119, 166.20730591, 128.70788574, 178.04029846]])}]
# 在程序中调用时，注释下一句 img = Image.open(img)，直接将图片输入到output函数中即可
'''
#print(output(Image.open('/media/xuer/Seagate Slim Drive/camera_raw_morning_0814/10000/camera_raw_left/12-2020-08-14_09_56_11.jpg')))
# position, label = output((Image.open('/media/xuer/Seagate Slim Drive/camera_raw/8000_exposure/0-2020-08-09_21_29_05.jpg')))
# print(position, label)