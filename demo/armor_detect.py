# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:41:42 2020

@author: Administrator
"""
import cv2
import numpy as np
import math

show_detail = False

def read_morphology(cap):  # read cap and morphological operation to get led binary image.
    ret, frame = cap.read()
    size_img = frame.shape[:2]
    #WIDTH, HIGH = int(570), int(600)
    factor = round(600/size_img[0])
    WIDTH, HIGH = int(size_img[1] * factor), int(size_img[0] * factor)

    frame = cv2.resize(frame, (WIDTH, HIGH), interpolation=cv2.INTER_CUBIC)

    mask = hsv_change(frame)
    dst_open = open_binary(mask, 3, 3)
    #dst_close = close_binary(mask, 3, 3)
    dst_erode = erode_binary(dst_open, 3, 3)
    #dst_erode = erode_binary(dst_close, 3, 3)
    dst_dilate = dilate_binary(dst_erode, 3, 3)
    cv2.circle(frame, (int(WIDTH / 2), int(HIGH / 2)), 2, (255, 0, 255), -1)

    cv2.imshow("mask", mask)
    cv2.imshow("dilate",dst_dilate)
    cv2.moveWindow("dilate", 0, 600)

    return dst_dilate, frame

def read_morphology_temp(frame):  # read cap and morphological operation to get led binary image.
    size_img = frame.shape[:2]
    
    #WIDTH, HIGH = int(570), int(600)
    factor = round(600/size_img[0])
    WIDTH, HIGH = int(size_img[1] * factor), int(size_img[0] * factor)

    frame = cv2.resize(frame, (WIDTH, HIGH), interpolation=cv2.INTER_CUBIC)

    mask = hsv_change(frame)
    dst_open = open_binary(mask, 13, 13)
    #dst_close = close_binary(mask, 3, 3)
    dst_erode = erode_binary(dst_open, 3, 3)
    #dst_erode = erode_binary(dst_close, 3, 3)
    dst_dilate = dilate_binary(dst_erode, 3, 3)
    cv2.circle(frame, (int(WIDTH / 2), int(HIGH / 2)), 2, (255, 0, 255), -1)

    if show_detail:
        cv2.imshow("mask", mask)
        cv2.imshow("dilate",dst_dilate)
        cv2.moveWindow("mask", 0, 0)
        cv2.moveWindow("dilate", 500, 0)

    return dst_dilate, frame, factor

def hsv_change(frame):  # hsv channel separation.

    # gray = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_hsv = np.array([0,0,130])
    #upper_hsv = np.array([120,255,255])
    lower_hsv = np.array([0,0,160])
    upper_hsv = np.array([255,255,255])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    return mask

def open_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return dst


def close_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return dst


def erode_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.erode(binary, kernel)
    return dst


def dilate_binary(binary, x, y):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
    dst = cv2.dilate(binary, kernel)
    return dst

def find_contours(binary, frame,rect_index):  # find contours and main screening section
    contour = []
    C_lights = []

##--------------------------------1. find contours-----------------------------##
    contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if show_detail:
        frame = cv2.drawContours(frame,contours,-1,(0,0,255),3)
    
##--------------------------------2. fit ellipse and RECT rcBound--------------##
##judge by calculate solidity of contours and area.
    for i in range(len(contours)):
        if len(contours[i]) >=5:
            ellipse=cv2.fitEllipse(contours[i])    #用一个椭圆来匹配目标。它返回一个旋转了的矩形的内接椭圆
            (x, y), (MA, ma), angle = ellipse[0], ellipse[1], ellipse[2]
            ellipseArea = 3.14 * MA * ma/4         #计算外接椭圆面积
            area = cv2.contourArea(contours[i])    #计算轮廓面积
            if area/ellipseArea >=0.8 and area >= 300 and area <= 4000:     #判断凸度(Solidity) 和轮廓面积大小
                if show_detail:
                    frame=cv2.ellipse(frame,ellipse,(0,255,0),2)
                    #print(area)
                rect = cv2.minAreaRect(contours[i])   #根据轮廓得到外接矩阵 长宽不固定，靠近x轴定义长
                if rect[2] < -45:
                    if rect[1][0]/rect[1][1] > 0.8:      #判断矩阵长宽比？？？
                        #a=tuple(((rect[1][0]*1.0,rect[1][1])))      #外接矩形(宽，长)？？？
                        #rect = tuple((rect[0],a,rect[2]))
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        if show_detail:
                            frame = cv2.drawContours(frame,[box],0,(255,0,0),2)
                            print(area)
                        contour.append((rect,box,area,x,y))   #保存符合要求的外接矩形及四点坐标，面积和中心x坐标
                elif rect[2] > -45:
                    if rect[1][1]/rect[1][0] > 0.8:      #判断矩阵长宽比？？？
                        #a=tuple(((rect[1][0]*1.0,rect[1][1])))      #外接矩形(宽，长)？？？
                        #rect = tuple((rect[0],a,rect[2]))
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        if show_detail:
                            frame = cv2.drawContours(frame,[box],0,(255,0,0),2)
                            print(area)
                        contour.append((rect,box,area,x,y))   #保存符合要求的外接矩形及四点坐标，面积和中心x坐标

##-----------------------------3. Find the right pair of lights----------------##
##campare each pair of contours, The approximate parallel, length-width ratio

    if len(contour) >=2:
        for i in range(len(contour)):
            j = i+1
            while j < len(contour):
                if contour[i][0][2] > -45:      #外接矩形定义：与碰到的矩形的第一条边的夹角。并且这个边的边长是width，另一条边边长是height。对旋转角修正
                    orientation_i = contour[i][0][2] - 90
                else:
                    orientation_i = contour[i][0][2]
                if contour[j][0][2] > -45:
                    orientation_j = contour[j][0][2] - 90
                else:
                    orientation_j = contour[j][0][2]
                if abs(orientation_i-orientation_j) < 30.0:   #判断是否平行
                    if contour[i][2]/contour[j][2] >= 0.4 and contour[i][2]/contour[j][2] <= 2.5:           #判断矩形面积比
                        #if abs(contour[i][3]-contour[j][3]) > 50 and abs(contour[i][4]-contour[j][4])<100 and abs(contour[i][3]-contour[j][3]) < 150:
                        if abs(contour[i][3]-contour[j][3]) > 50 and abs(contour[i][4]-contour[j][4])<100:
                            if show_detail:
                                frame = cv2.drawContours(frame,[contour[i][1]],0,(255,0,0),2)
                                frame = cv2.drawContours(frame,[contour[j][1]],0,(0,255,0),2)
                            couple_light = tuple((contour[i],contour[j]))
                            C_lights.append(couple_light)
                        
                j +=1

##----------------------------4. Find best pair of lights----------------------##
    
    if len(C_lights) > 1:
        distance = []
        for i in range(len(C_lights)):
            d = np.sqrt((C_lights[i][0][3] - C_lights[i][1][3])**2 + (C_lights[i][0][4] - C_lights[i][1][4])**2  )    #找到两个x坐标最相近的灯条
            distance.append(d)
            if show_detail:
                print(d)
        index = np.argmin(distance)
    elif len(C_lights) == 1:
        index = 0
    else:
        return 0,[0]
    if C_lights[index][0][0][1][0] >= C_lights[index][0][0][1][1] and C_lights[index][1][0][1][0] >= C_lights[index][1][0][1][1]:  #判断旋转角是否异常
        angle = (C_lights[index][0][0][2] + C_lights[index][1][0][2])/2
        l_light = cv2.boxPoints(tuple((C_lights[index][0][0][0],C_lights[index][0][0][1],angle)))
        r_light = cv2.boxPoints(tuple((C_lights[index][1][0][0],C_lights[index][1][0][1],angle)))
    else:
        l_light = C_lights[index][0][1]
        r_light = C_lights[index][1][1]
    if show_detail:
        frame = cv2.drawContours(frame,[np.int0(l_light)],0,(255,0,0),2)      #画出两个灯条矩形
        frame = cv2.drawContours(frame,[np.int0(r_light)],0,(0,255,0),2)


##---------------------------5. Find Armor through pair of lights--------------##

    n = len(l_light) + len(r_light)
    cnt = np.zeros((n,1,2))
    for i in range(len(l_light)):
        cnt[i][0] = l_light[i]
    for j in range(len(r_light)):
        cnt[len(l_light)+j][0] = r_light[j]
    cnt = cnt.astype(int)
    Rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(Rect)
    box = np.int0(box)
    if show_detail:
        frame = cv2.drawContours(frame,[box],0,(0,0,255),2)           #画出装甲板矩形
        #robot = cv2.drawContours(robot,[np.int0(box/factor)],0,(0,0,255),2)
        filename = 'armor_' + str(rect_index)           
        cv2.imshow(filename,frame)
        #cv2.imshow('a',robot)
        if rect_index == 0:
            cv2.moveWindow(filename, 0, 1000)
        elif rect_index == 1:
            cv2.moveWindow(filename, 600, 1000)
        elif rect_index ==2:
            cv2.moveWindow(filename, 1200, 1000)
        elif rect_index ==3:
            cv2.moveWindow(filename, 1800, 1000)
    return C_lights,box

if __name__ == '__main__':
    #cap = cv2.VideoCapture("video_footage/img1.jpg") 
    #cap = cv2.VideoCapture("video_footage/8000_exposure_robot/127.jpg")
    cap = cv2.VideoCapture("video_footage/20200814_10000_robot/287.jpg")
    dst_dilate, frame = read_morphology(cap)
    C_lights,Rect = find_contours(dst_dilate,frame,0)
