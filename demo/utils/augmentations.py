import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A  # 数据增强
import torchvision.transforms as T

# boxes = (cls, x, y, w, h)
def horizontal_flip(images, boxes):
    images = np.flip(images, [-1])
    boxes[:, 1] = 1 - boxes[:, 1]
    return images, boxes

# images[np.unit8], boxes[numpy] = (cls, x, y, w, h)
def augment(image, boxes):  # 输入的是填充后的图片, 填充图片下的xywh比例坐标
    h, w, _ = image.shape
    labels, boxes_coord = boxes[:, 0], boxes[:, 1:]
    labels = labels.tolist()  # boxes[:, 0],label的id
    # boxes_coord = boxes_coord * h     # 得到填充图片尺寸下的坐标（未归一化的坐标）(实际坐标)
    boxes_coord[:, 0] *= w
    boxes_coord[:, 1] *= h
    boxes_coord[:, 2] *= w
    boxes_coord[:, 3] *= h
    boxes_coord[:, 0] = np.clip(boxes_coord[:, 0]-boxes_coord[:, 2]/2, a_min=0, a_max=None)  # 确保x_min和y_min有效  # clip:限制最大最小值
    boxes_coord[:, 1] = np.clip(boxes_coord[:, 1]-boxes_coord[:, 3]/2, a_min=0, a_max=None)
    boxes_coord = boxes_coord.tolist()  # [x_min, y_min, width, height]
    # x_min, y_min, width, height 注意数据增强时check又用的左上角坐标(coco格式)
    # 在这里设置数据增强的方法
    # aug = T.Compose([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=10, hue=0.1)])
    aug = A.Compose([  # 需修改filter_bbox的min_area
        A.HorizontalFlip(p=0.5),  # 依概率p水平翻转：transforms.RandomHorizontalFlip(p=0.5)
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=10, val_shift_limit=10, p=1),  # HSV 色调（H）饱和度（S）明度（V） 参数为变化的范围(-limit, +limit)  yolo中V和S为1到1.5倍 H为-0.1~0.1
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=1),  # 亮度对比度
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=5, border_mode=0, p=0.5),  # 移动放缩旋转(yolo中为-5~5度)(-shift_limit, +shift_limit) (-scale_limit+1, +scale_limit+1) (-rotate_limit, +rotate_limit)
        # A.RandomCrop(height=0.7*h, width=0.7*w, always_apply=False, p=0.5),  # 随机裁剪 参数为裁剪后的w和h 起始为左上角原点
        # A.Rotate(limit=5, p=1),  # 旋转 参数为旋转角度范围(-limit, limit)
        A.RandomSizedCrop(min_max_height=(int(0.7*h), h), height=h, width=w, w2h_ratio=1.7)  # 放缩裁剪  min_max_height为裁剪后的height,在0.7h和h之间随机取值,裁剪后的宽为1.7h width=w为裁剪后放缩到这样大小 w2h_ratio为裁剪后宽高比
    ], bbox_params={'format': 'coco', 'label_fields': ['category_id']})  # 分bbox_params/keypoint_params
    # 实例化compose类
    #　def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0):
    # super(Compose, self).__init__([t for t in transforms if t is not None], p)
    # 继承class BaseCompose(object):
    # def __init__(self, transforms, p):(通过循环方式将transforms中的增强方式都传给compose继承的父类BaseCompose)
    # print(aug.p) 依然是默认值1.0
    #

    augmented = aug(image=image, bboxes=boxes_coord, category_id=labels)
    # HorizontalFlip中的参数(apply中的image bbox)
    # 调用def __call__(self, force_apply=False, **data):
    # (transform在__call__中实现(for idx, t in enumerate(transforms):))
    # 实现过程是:先利用convert_keypoints_to_albumentations函数将读入的bbox坐标信息由coco格式(不是yolo格式!)转化为xyxy比例坐标格式,并check掉超出1的和max>min的, 然后再进行数据增强操作(img和bbox)
    # 真正进入transform的是xyxy格式的


    if augmented['bboxes']:  # 将aug之后的boxes由xminyminxy实际坐标格式转化为xywh比例坐标格式
        image = augmented['image']
        boxes_coord = np.array(augmented['bboxes'])  # x_min, y_min, w, h → x, y, w, h
        boxes_coord[:, 0] = boxes_coord[:, 0] + boxes_coord[:, 2]/2
        boxes_coord[:, 1] = boxes_coord[:, 1] + boxes_coord[:, 3]/2
        # boxes_coord = boxes_coord / h
        boxes_coord[:, 0] /= w
        boxes_coord[:, 1] /= h
        boxes_coord[:, 2] /= w
        boxes_coord[:, 3] /= h
        labels = np.array(augmented['category_id'])[:, None]
        boxes = np.concatenate((labels, boxes_coord), 1)
    else:  # 经过aug之后，如果把boxes变没了，则返回原来的图片
        image = image
        boxes[:, 1] /= w
        boxes[:, 2] /= h
        boxes[:, 3] /= w
        boxes[:, 4] /= h
    return image, boxes