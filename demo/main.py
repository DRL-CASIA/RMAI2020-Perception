##v4版本可以识别多车并对应,并行运算
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:53:40 2020

@author: Administrator
"""
import sys
sys.path.append("./angle_classify")
sys.path.append("./armor_classify")
sys.path.append("./car_classify")
import numpy as np
import cv2
from armor_detect_withlightbox import read_morphology_withlightbox,find_contours_withlightbox
from armor_detect import read_morphology_temp,find_contours
from yolo_detect_v2 import output
from position_predict import *
from utils.utils_mulanchor import *
import torch
from models_nolambda_focallossw import *
import time
from classification import *
from classification_car import *
from classification_angle_camera import *
from multiprocessing.dummy import Pool as ThreadPool


camera = 'left'

def camera_calibration(img,camera='left'):
    # # TODO 获取相机内参，获取二维码四点世界坐标
    np.set_printoptions(suppress=True)
    object_3d_points = np.array(([-75, -75, 0],
                                 [75, -75, 0],
                                 [75, 75, 0],
                                 [-75, 75, 0]), dtype=np.double)
    # TODO 将 object_2d_point 的值设为 detect得到的二维码四点坐标
    object_2d_point = np.array(([954., 534.],
                                [1004., 536.],
                                [1006., 579.],
                                [956., 577.]), dtype=np.double)
    if camera == 'left':
        camera_matrix = np.array([[6.570931846420799e+02,0,3.196939147616254e+02],
                                  [0,6.190714811365291e+02,2.520205008433231e+02],
                                  [0,0,1]], dtype="double")
        dist_coeffs = np.transpose([-0.216248222896496, 0.226313370014235, -0.001139415943532, 
                                    -0.004624035593808, -0.059067986510048])
    
    if camera == 'right':
        camera_matrix = np.array([[653.528968471312,0,316.090142900466],
                                  [0,616.850241871879,242.354349211058],
                                  [0,0,1]], dtype="double")
        dist_coeffs = np.transpose([-0.203713353732576, 0.178375149377498, -0.000880727909602325, 
                                    -0.00023370151705564, -0.0916209128198407])
    found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coeffs)
    rotM = cv2.Rodrigues(rvec)[0]
    return np.array(rotM).T, np.array(tvec)
    

def point_sort(box):
    x = [box[0][0],box[1][0],box[2][0],box[3][0]]
    index = np.argsort(x)
    left = [box[index[0]],box[index[1]]]
    right = [box[index[2]],box[index[3]]]
    if left[0][1]< left[1][1]:
        left_up = left[0]
        left_down = left[1]
    else:
        left_up = left[1]
        left_down = left[0]
    if right[0][1]< right[1][1]:
        right_up = right[0]
        right_down = right[1]
    else:
        right_up = right[1]
        right_down = right[0]
    return left_up,left_down,right_up,right_down

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

def draw_position_rect(im, left_up,left_down,right_up,right_down):
    #   原理是：：：PNP算法
    #   找到四个对应点，根据摄像头参数求解实际世界坐标
    #   找外接矩形的四个图像点
    #   分别设置为（0，0，0），（0，车体长度，0），（0，车体长度，车体高度），（0，0，车体高度）///
    #   但是这样做不对，因为车体在旋转过程中无法在图像上找到精确的位置，无法计算。
    #   应该以检测装甲板的位置作为四个对应点，这样他的大小是死的，是可以计算的。“
    image_points = np.array([
        (left_up[0], left_up[1]),
        (right_up[0], right_up[1]),
        (right_down[0], right_down[1]),
        (left_down[0], left_down[1]),
    ], dtype="double")
    high = 60 #mm
    width = 137 #mm
    model_points = np.array([
        (-width/2, -high/2, 0),
        (width/2, -high/2, 0),
        (width/2, high/2, 0),
        (-width/2, high/2, 0),
    ])

    camera_matrix = np.array([[6.570931846420799e+02,0,3.196939147616254e+02],
                              [0,6.190714811365291e+02,2.520205008433231e+02],
                              [0,0,1]], dtype="double")
    dist_coeffs = np.transpose([-0.216248222896496, 0.226313370014235, -0.001139415943532, 
                                -0.004624035593808, -0.059067986510048])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points, camera_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    rotationtion_vector = cv2.Rodrigues(rotation_vector)[0]
    distance = np.sqrt(translation_vector[0]**2+translation_vector[1]**2+translation_vector[2]**2)
   
    return rotationtion_vector, translation_vector,distance/1000

def armor_6(fig):   

    array = fig
    fig = cv2.resize(array,(48, 48))
    fig = torch.Tensor(fig)
    fig = fig.permute((2,0,1))
    img = torch.unsqueeze(fig, 0)
    outputs = net_model(img.cuda())
    _, predicted = torch.max(outputs.data, 1)

    return int(predicted)

def car_6(fig):    
    array = fig
    fig = cv2.resize(array,(56,56))
    fig = torch.Tensor(fig)
    fig = fig.permute((2,0,1))
    img = torch.unsqueeze(fig, 0)
    outputs = net_model_car(img)
    _, predicted = torch.max(outputs.data, 1)

    return int(predicted)

def world_angle_6(fig, pose,camera = 'left'):

    pose_array = pose

    pose_x = pose_array[0]
    pose_y = pose_array[1]
    pose_x = float(pose_x)
    pose_y = float(pose_y)
    pose_array = (pose_x, pose_y)
    pose_array = np.array(pose_array, dtype='float').reshape(1,2)
    pose_array = torch.tensor(pose_array)

    array = fig
    fig = cv2.resize(array, (56, 56))
    fig = torch.Tensor(fig)
    fig = fig.permute(2, 0, 1)
    img = torch.unsqueeze(fig, 0)
    outputs = net_model_angle(img.cuda(), pose_array.cuda())
    _, predicted = torch.max(outputs.data, 1)
    
    predicted = int(predicted)
    # 坐标转换
    pi = math.pi
    alpha = 0
    di = pi / 8
    theta = di * (2 * predicted + 1)
    try:
        if (theta >= pi / 2 + math.atan(pose_x / pose_y) and theta < pi):
            alpha = theta - pi / 2 - math.atan(pose_x / pose_y)
        elif(theta >= pi * 2 - math.atan(pose_y / pose_x) and theta < pi * 2):
            alpha = theta - pi * 3 + math.atan(pose_y / pose_x)
        else:
            alpha = theta - pi + math.atan(pose_y / pose_x)
    except:
        pass
    return alpha, predicted

cap = cv2.VideoCapture("video_footage/1cars.avi")

if (cap.isOpened() == False):
    print("Error opening video stream or file")
position_data = []
n =0
frame_id = 0


#-----------yolo model------------------#
cfgfile = "cfg/yolov3_camera_raw_3_pre_resprune_sp0.001_p0.01_sp0.001_p0.01.cfg"
weightsfile = "cfg/yolov3_camera_raw_3_pre_resprune_sp0.001_p0.01_sp0.001_p0.01.weights"
names =  "cfg/camera_raw_0817_3.names"
classes = load_classes(names)
num_classes = 2

start = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = torch.cuda.is_available()
inp_dim = [416,416]
bbox_attrs = 5 + num_classes

print("Loading network.....")
model = Darknet(cfgfile, inp_dim).to(device)
model.load_darknet_weights(weightsfile)



#--------------------------distance nihe------------------------#
mlp_model = load_mlp_model(camera)
mlp_model.eval()
print("Network successfully loaded")


#-----------------------class model---------------------------#
net_model = classification_modelload()
net_model_car = car_classification_modelload()
#-----------------------anger model---------------------------#
net_model_angle = classification_angle_camer_modelload(camera)


if CUDA:
    model.cuda()
    mlp_model.cuda()
    
model(get_test_input(inp_dim, CUDA)).to(device)
model.eval().to(device)

log_path = './video_footage/20200824/log.log'   #读取车位姿信息

f=open(log_path,"r")
lines = f.readlines()
f.close()

ret, frame = cap.read()
rotationtion_vector_cam,translation_vector_cam = camera_calibration(frame,'left')

time_start = time.time()    
while (cap.isOpened()):
    try:
        ret, frame = cap.read()
        size_img = frame.shape[:2]
        frame_show = frame.copy()
        x_r,y_r = float(lines[(frame_id+1)*2].split(' ')[1]),float(lines[(frame_id+1)*2].split(' ')[2]) 
        d_r = np.sqrt(x_r**2+y_r**2+2.07**2)         #每张图对应车位置
        frame_id += 1
    except:
        print('time cost:', time_stop-time_start)
        break
    if ret == True:
        t_start = time.time()
        output_dict = output(frame, CUDA, model,device,num_classes)
        #t_yolo = time.time()
        for i in range(len(output_dict)):
            #global n,frame_show
            light = 0
            output_dict[i]['img_id'] = []
            output_dict[i]['car_class'] = []
            output_dict[i]['car_angle'] = []
            output_dict[i]['light_box'] = np.zeros((len(output_dict[i]['armor_box'])+1,4,2))
            output_dict[i]['position'] = np.zeros((len(output_dict[i]['armor_box'])+1,2))
            if len(output_dict[i]['armor_box']) != 0:
                y0,h = int(round(output_dict[i]['armor_box'][0][1]))-5,int(round(output_dict[i]['armor_box'][0][3])) - int(round(output_dict[i]['armor_box'][0][1]))+10
                x0,w = int(round(output_dict[i]['armor_box'][0][0]))-5,int(round(output_dict[i]['armor_box'][0][2])) - int(round(output_dict[i]['armor_box'][0][0]))+10
                robot = frame[y0:y0+h,x0:x0+w]
                if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                    car_class = armor_6(robot)
                    output_dict[i]['car_class'] = car_class
                for j in range(len(output_dict[i]['armor_box'])):

                    index = j
                    y0,h = int(round(output_dict[i]['armor_box'][j][1]))-5,int(round(output_dict[i]['armor_box'][j][3])) - int(round(output_dict[i]['armor_box'][j][1]))+10
                    x0,w = int(round(output_dict[i]['armor_box'][j][0]))-5,int(round(output_dict[i]['armor_box'][j][2])) - int(round(output_dict[i]['armor_box'][j][0]))+10
                    robot = frame[y0:y0+h,x0:x0+w]
                    n +=1
                    if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                        dst_dilate,robot_resize, factor = read_morphology_withlightbox(robot)
                        #cv2.rectangle(frame_show, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 1)
                        _, box = find_contours_withlightbox(dst_dilate,robot_resize,index)
                        if len(box) != 1:
                            time_calculate1 = time.time()
                            light += 1
                            for l in range(len(box)):
                                box[l][0] = box[l][0]/factor + x0
                                box[l][1] = box[l][1]/factor + y0
                            box = np.int0(box)
                            frame_show = cv2.drawContours(frame_show,[box],0,(0,0,255),2)
                            left_up,left_down,right_up,right_down = point_sort(box)
                            print('%d.jpg'%(frame_id))
                            if frame_id == 258:
                                break
                            rotationtion_vector, translation_vector,distance = draw_position_rect(frame_show, left_up,left_down,right_up,right_down )
                            #-------from Camera coordinate system to world coordinate system-----#

                            position_world = np.dot(np.linalg.inv(rotationtion_vector_cam),(translation_vector-translation_vector_cam))
                            #print(position_world)
                            x = (position_world[2] + 3260)/1000
                            y = (-position_world[0] + 440)/1000+0.3
                            output_dict[i]['light_box'][j] = box
                            output_dict[i]['position'][j] = (x,y)

                        if np.sqrt(((x0+w/2)-257)**2+((y0+h/2)-220)**2) > 50:
                            cv2.rectangle(frame_show, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 1)
            elif len(output_dict[i]['armor_box']) == 0 or light == 0:
                    y0,h = int(round(output_dict[i]['car_box'][1]))-5,int(round(output_dict[i]['car_box'][3])) - int(round(output_dict[i]['car_box'][1]))+10
                    x0,w = int(round(output_dict[i]['car_box'][0]))-5,int(round(output_dict[i]['car_box'][2])) - int(round(output_dict[i]['car_box'][0]))+10
                    robot = frame[y0:y0+h,x0:x0+w]
                    if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                        car_class = car_6(robot)

                    n +=1
                    if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:
                        dst_dilate, robot_resize, factor = read_morphology_temp(robot)
                        #cv2.rectangle(frame_show, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 1)
            
                        _, box = find_contours(dst_dilate,robot_resize,0)
                        if len(box) != 1:
                            for l in range(len(box)):
                                box[l][0] = box[l][0]/factor + x0
                                box[l][1] = box[l][1]/factor + y0
                            box = np.int0(box)
                            #frame_show = cv2.drawContours(frame_show,[box],0,(0,0,255),2)
                            left_up,left_down,right_up,right_down = point_sort(box)
                            print('%d.jpg'%(frame_id))
                            rotationtion_vector, translation_vector,distance = draw_position_rect(frame_show, left_up,left_down,right_up,right_down )
                            #-------from Camera coordinate system to world coordinate system-----#

                            position_world = np.dot(np.linalg.inv(rotationtion_vector_cam),(translation_vector-translation_vector_cam))

                            x = (position_world[2] + 3260)/1000
                            y = (-position_world[0] + 440)/1000+0.3
                            output_dict[i]['position'][-1] = (x,y)

        
            # -------------MLP 位置预测 --------------------------------#
            if 'car_box' in output_dict[i]:
                time_positionpre1=time.time()
                mlp_x,mlp_y=position_prediction(mlp_model,output_dict[i]['car_box'])
                output_dict[i]['position_mlp'] = [mlp_x, mlp_y]
                time_positionpre2=time.time()

                # fusion
                position_f = position_fusion(output_dict[i])
                output_dict[i]['position_fusion'] = position_f
            # ------------angle predicted------------------------------#
            if len(output_dict[i]['car_box']) != 0 :
                y0,h = int(round(output_dict[i]['car_box'][1])),int(round(output_dict[i]['car_box'][3])) - int(round(output_dict[i]['car_box'][1]))
                x0,w = int(round(output_dict[i]['car_box'][0])),int(round(output_dict[i]['car_box'][2])) - int(round(output_dict[i]['car_box'][0]))
                robot = frame[y0:y0+h,x0:x0+w]
                if np.shape(robot)[0] !=0 and np.shape(robot)[1] !=0:

                    pose = output_dict[i]['position_mlp']
                    angle, predicted = world_angle_6(robot, pose)
                    output_dict[i]['car_angle'] = angle
                    time_anglepre2=time.time()
                car_class_dict = ['blue-1','blue-2','red-1','red-2','grey-1','grey-1']
                try:
                    if len(output_dict[i]['armor_box']) != 0:
                        cv2.rectangle(frame_show, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 1)
                        text = 'ID: ' + car_class_dict[output_dict[i]['car_class']] + ' Pose: ' + str(round(output_dict[i]['car_angle'], 3))
                        cv2.putText(frame_show, text, (x0,y0-20), cv2.FONT_HERSHEY_PLAIN, 1, [0,255,0], 1)


                        final_out = output_dict[i]['position_mlp']
                        frame_show = cv2.putText(frame_show, 'x=%.2f,y=%.2f' % (final_out[0],final_out[1]), (x0,y0),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                except:
                    pass

        t_stop = time.time()
        time_stop = time.time()
        print('t cost:', t_stop - t_start)
        #-----------test log------------------------------#
        '''text = 'car_1 x: ' + str(x_r) + ' car_1 y: ' + str(y_r)
        cv2.putText(frame_show, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 1, [0,255,0], 1)
        text = 'car_2 x: ' + str(x_r1) + ' car_2 y: ' + str(y_r1)
        cv2.putText(frame_show, text, (50,70), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,255], 1)'''
        img_name = str(frame_id) + '.jpg'
        img_p = './fig4/' + img_name

        cv2.imshow('img', frame_show)
        cv2.imwrite(img_p,frame_show)
        cv2.moveWindow('img', 0, 0)    
        cv2.waitKey(5)

    else:
        print('time cost:', time_stop-time_start)
        break
