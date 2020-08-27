from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(threshold=np.inf)
from utils.parse_config import *
from utils.utils_mulanchor import build_targets, to_cpu, non_max_suppression
from utils.focal_loss import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# convolutional,maxpool,upsample,route,shortcut,yolo

def create_modules(module_defs):
    resnum = 0
    # [{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},...]
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # netinfo  pop() 函数用于移除列表中的一个元素,并且返回该元素的值。
    output_filters = [int(hyperparams["channels"])]  # 3
    module_list = nn.ModuleList()       # 一定要用ModuleList()才能被torch识别为module并进行管理，不能用list！
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()  #  一个时序容器 Modules 会以他们传入的顺序被添加到容器中  # 即每个小modules都是一个时序容器,例如卷积层包括sequencial(conv,bn,relu)

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),  # 定义Conv2d的属性  add_module(name, module) 在已有module上加上刚创建的卷积层，name中{module_i}将被module_i代替，即index的值
            )
            if bn:
                # modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters))  # 定义BatchNorm2d的属性
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1, inplace=True))  # 定义LeakyReLU的属性

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]  # 一个或两个元素
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
        # 创建的空的层 用来实现此处的添加 这里用之前创建的空层来占位，之后再定义具体的操作 对应的操作在class darknet中定义

        elif module_def["type"] == "shortcut":
            resnum += 1
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]  # 读取这个yolo层的mask
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  # 每两个取一位
            anchors = [anchors[i] for i in anchor_idxs]  # 格式为[a,b]
            num_classes = int(module_def["classes"])
            # img_size = int(hyperparams["height"])
            img_size = [int(hyperparams["width"]), int(hyperparams["height"])]  # 读取cfg
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)  # 此处定义yololayer的输入参量(锚点/类别数/尺度), 定义yolo层
            modules.add_module(f"yolo_{module_i}", yolo_layer)  # 将一个 child module 添加到当前 modle
        # Register module list and number of output filters
        module_list.append(modules)  # 将创建的module添加进modulelist中 # modules是根据cfg顺序创建的
        output_filters.append(filters)  # filter保存了输出的维度
        # sequential和modulelist的区别在于后者需要定义forward函数

    return hyperparams, module_list, resnum


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)  # 上采样的插值方式
        return x


class EmptyLayer(nn.Module):    # 只是为了占位，以便处理route层和shortcut层
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors  # 读取的cfg文件设置
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()  # yolo层的子层  均方误差 损失函数，计算 检测时的坐标损失
        self.bce_loss = nn.BCELoss()  # yolo层的子层  计算目标和输出之间的二进制交叉熵  损失函数，计算  多类别的分类损失
        self.obj_scale = 1
        self.noobj_scale = 50
        self.metrics = {}
        self.img_dim = img_dim  # w608 h352
        self.grid_size_w = 0
        self.grid_size_h = 0  # grid size

    def compute_grid_offsets(self, grid_size_w, grid_size_h,cuda=True):
        # Calculate offsets for each grid 计算每个网格的偏移量
        # torch.linspace(arange)返回 start 和 end 之间等间隔 steps 点的一维 Tensor
        # repeat沿着指定的尺寸重复 tensor
        # 过程：
        #      torch.linspace(0, g_dim-1, g_dim)  ->  [1,13]的tensor
        #      repeat(g_dim,1)                    ->  [13,13]的tensor 每行内容为0-12,共13行
        #      repeat(bs*self.num_anchors, 1, 1)  ->  [48,13,13]的tensor   [13,13]内容不变，在扩展的一维上重复48次
        #      view(x.shape)                      ->  resize成[16.3.13.13]的tensor
        # grid_x、grid_y用于 定位 feature map的网格左上角坐标

        # self.grid_size = grid_size
        self.grid_size_w = grid_size_w  # 如19
        self.grid_size_h = grid_size_h  # 如11
        # g = self.grid_size
        g = [self.grid_size_w, self.grid_size_h]
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim[0] / self.grid_size_w
        # self.stride_w = self.img_dim[0] / self.grid_size_w  # 每个grid的宽 608/19
        # self.stride_h = self.img_dim[1] / self.grid_size_h  # 每个grid的高 352# self.stride = self.img_dim / self.grid_size/11
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g[0]).repeat(g[1], 1).view([1, 1, g[1], g[0]]).type(FloatTensor)# xy定位左上角坐标(grid索引坐标)  view: 行数 列数  # x方向 列坐标
        """print(self.grid_x)
        [[[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.],
          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.]]]]"""
        # eg:[1,g[0]]的tensor -> [g[1],g[0]]的tensor -> [1,1,g[0],g[1]]的tensor
        self.grid_y = torch.arange(g[1]).repeat(g[0], 1).t().view([1, 1, g[1], g[0]]).type(FloatTensor)  # y方向 行坐标
        """print(self.grid_y)
        tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.],
          [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.],
          [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
            2.,  2.,  2.,  2.,  2.],
          [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
            3.,  3.,  3.,  3.,  3.],
          [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
            4.,  4.,  4.,  4.,  4.],
          [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
            5.,  5.,  5.,  5.,  5.],
          [ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,
            6.,  6.,  6.,  6.,  6.],
          [ 7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,
            7.,  7.,  7.,  7.,  7.],
          [ 8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,  8.,
            8.,  8.,  8.,  8.,  8.],
          [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
            9.,  9.,  9.,  9.,  9.],
          [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
           10., 10., 10., 10., 10.]]]])"""
        # 多加了一个转置(竖列)  # cfg中的anchor是针对预定的img_dim来的,在此处是352*608
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])  # anchor的尺度除以每一个grid的宽高 维度为[ self.num_anchors,2] 即每一个anchor占据了几个grid 即按输出特征图大小缩放后的anchor
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))  # anchor_w的范围是[0, grid_size](416下),浮点型数值
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))  # anchor占了几个预测框
        """print(self.anchor_w)
        print(self.anchor_h)
        tensor([[[[ 3.6250]],

         [[ 4.8750]],

         [[11.6562]]]])
         tensor([[[[ 2.8125]],

         [[ 6.1875]],

         [[10.1875]]]])"""

    def forward(self, x, targets=None, img_dim=None):  # 416*416为高乘以宽的tensor
        # x[batch, 3*(5+classes)(即最后一个卷积层的filter数), feature_map_h(如11 其实到最后就是grid的个数), feature_map_w(如19)]
        # x是最后一次卷积的结果
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim  # Darknet的forward中, img_dim = [x.shape[3], x.shape[2]] w608 h352 compute_grid_offsets

        num_samples = x.size(0)  # 图片的个数
        # grid_size = x.size(2)
        grid_size_w = x.size(3)  # feature map的宽度 如19 gird的横向个数
        grid_size_h = x.size(2)  # feature map的高度 如11 grid的纵向个数


        prediction = (  # 图片个数 锚点框个数 每个grid需预测的值个数 grid_h grid_w
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size_h, grid_size_w)
            .permute(0, 1, 3, 4, 2)     # num_samples, num_anchors, grid_size_h, grid_size_w, self.num_classes + 5
            .contiguous()               # contiguous返回一个内存连续的有相同数据的 tensor
        )
        # 每张图片中含有grid_w*grid_h个grid,每个grid中含有num_anchors个anchor,每个anchor负责num_classes+5个预测值

        # Get outputs  # 在最后一维self.num_classes + 5中[0,1,2,3]为预测的框偏移(tx/ty/tw/th), 4物体置信度（是否有物体）, 5：为多类别的分类概率
        x = torch.sigmoid(prediction[..., 0])  # Center x  省略号包括的维度为[num_samples, num_anchors, grid_size_h, grid_size_w]
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf 卷积然后sigmoid就可以得到偏移,conf和概率?
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        # pred_cls = prediction[..., 5:] # Cls pred.

        # If grid size does not match current we compute new offsets
        # if grid_size != self.grid_size:     # 不用每次都计算，只有在输入图片大小第一次发生变化时计算
        #     self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        if (grid_size_w != self.grid_size_w) | (grid_size_h != self.grid_size_h):     # 不用每次都计算，只有在输入图片大小第一次发生变化时计算
            self.compute_grid_offsets(grid_size_w, grid_size_h, cuda=x.is_cuda)

        # Add offset and scale with anchors
        # 预测框xywh更新
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # 生成形状与prediction[..., :4]相同的张量 ,每张图片的每个grid中包括x y w h
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y  # x.data是偏置, 实际上sigmoid的含义是使其范围限制在0~1之内,没有改变其偏置的本意
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # anchor_w的范围是[0,grid_size](416下)，浮点型变量
        # 这里的self.anchor_w是self.scaled_anchors, 即每个anchor占据的grid的个数, w的是指和tw一样,实际上这个运算过程是得出tw那个运算的逆运算
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,  # 每一张图片中的所有anchor(每个grid都加起来)都有xywh四个关于box的预测值  # 放缩到原图大小               # pred_boxes[..., 0].view(num_samples, -1, 1) * self.stride_w,     # 乘以grid的宽和高换算成真实坐标
                # pred_boxes[..., 1].view(num_samples, -1, 1) * self.stride_h,
                # pred_boxes[..., 2].view(num_samples, -1, 1) * self.stride_w,
                # pred_boxes[..., 3].view(num_samples, -1, 1) * self.stride_h,  # -1 = num_anchor * (3*(class+5))
                pred_conf.view(num_samples, -1, 1),  # 每一张图片中的所有anchor(每个grid都加起来)都有一个conf的预测值
                pred_cls.view(num_samples, -1, self.num_classes),
            ),  # 每一张图片中的所有anchor(每个grid都加起来)都有num_classs个关于类别的预测值
            -1,  #  cat维度: num_samples, num_anchors*grid_size*grid_size, num_classes+5
        )
        if targets is None:
            return output, 0  # 没有预先target，loss为0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # ------------新增均衡样本的becloss的weight------------------
            self.bce_clsloss = focal_BCELoss(alpha=2.5, gamma=2)  # [15,22]

            #  ------------------------------------------------------------
                        # 只有iou时用xyxy格式,其余时候均用xywh格式s
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # TODO：这里没有针对wh的损失进行加权处理
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  # 只计算标志位之下的loss
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])  # 计算偏置loss
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  # 计算置信度loss  # bce_loss = −1/n​∑(yn​×ln(xn)​+(1−yn​)×ln(1−xn​))  多分类损失
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])  # 索引一致 对应的值不一致
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_clsloss(pred_cls[obj_mask], tcls[obj_mask])  # 计算类别概率loss
            # print(tcls[obj_mask])
            # print(loss_cls)
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf * 15 + loss_cls * 50
            # total_loss = loss_x + loss_y + loss_w + loss_h + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()  # 预测正确的类别个数
            conf_obj = pred_conf[obj_mask].mean()  # 为1取该值 为0不取该值
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size_w": grid_size_w,
                "grid_size_h": grid_size_h,
            }

            return output, total_loss


class Darknet(nn.Module):  # 继承nn.module类 # 在子类进行初始化时，也想继承父类的__init__()就通过super()实现
    """YOLOv3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()  # 继承父类参数, 父类为nn.Module
        if isinstance(config_path, str):
            self.module_defs = parse_model_config(config_path)
        elif isinstance(config_path, list):
            self.module_defs = config_path  # 已经是modeldefs形式?

        [self.hyperparams, self.module_list, _] = create_modules(self.module_defs)  # modulelist是根据cfg创建的
        """print(self.module_list):
        ModuleList(
          (0): Sequential(
          (conv_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_0): LeakyReLU(negative_slope=0.1, inplace)
          )
          (1): Sequential(
          (conv_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_1): LeakyReLU(negative_slope=0.1, inplace)
          )
          ......
        )"""
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]  # layer是个nn.Sequential()  # 只有yolo层存在metric属性项
        # 加layer[0]的原因: nn.sequence
        self.img_size = img_size  # 修改为[608,352]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        # self.L = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.L.data.fill_(1)
        # self.register_parameter('Lambda', self.L)

    def forward(self, x, targets=None):
        # img_dim = x.shape[2]# 取决于输入图片的大小，因为是正方形输入，所以只考虑height
        img_dim = [x.shape[3], x.shape[2]]  # 此处有问题,img_dim实际上没有传入yolo_layer函数之中,传入的是cfg文件中的width和height; 此处传入到yolo_layer中的forward中
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            """
            print(module)各层形式如下:
            Sequential(
                (route_95): EmptyLayer()
            ) # yolo2以后输出从route层开始
            Sequential(
                (shortcut_40): EmptyLayer()
            )
            Sequential(
                (conv_80): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (batch_norm_80): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (leaky_80): LeakyReLU(negative_slope=0.1, inplace)
            )
            Sequential(
                (conv_81): Conv2d(1024, 33, kernel_size=(1, 1), stride=(1, 1))
            )
            Sequential(
              (yolo_82): YOLOLayer(
                (mse_loss): MSELoss()
                (bce_loss): BCELoss()
              )
            """
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)# 直接调用module.py中的class Module(object)的类的 result = self.forward(*input, **kwargs)的输入
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                # print(self.t)
                x = layer_outputs[-1] + layer_outputs[layer_i]
                # x = self.L * layer_outputs[-1] + layer_outputs[layer_i]
                # print(self.L)
                # print(self.A['a' + str(self.t)])
            elif module_def["type"] == "yolo":  # [82, 94, 106] for yolov3
                """print(module[0])
                YOLOLayer(
                  (mse_loss): MSELoss()
                  (bce_loss): BCELoss()
                )
                """
                # print("yolo!")
                x, layer_loss = module[0](x, targets, img_dim)  # module是nn.Sequential()，所以要取[0]
                # 如果去掉[0],result = self.forward(*input, **kwargs): TypeError: forward() takes 2 positional arguments but 4 were given (*args表示任何多个无名参数，它是一个tuple, **kwargs表示关键字参数，它是一个dict)
                # 加不加[0]的区别在于加[0]先调用container.py中的class Sequential(Module)类,取得module.py中的 class Module(object)的类的 result = self.forward(*input, **kwargs)的输入;不加[0]则直接调用*input, **kwargs
                # 此处输入的是yololayer中forward的输入参量
                # x为以上层的输出（yolo层的是输入），target为目标值  该层yolo_layer函数的定义(_init_函数)已由createmodels函数完成
                loss += layer_loss
                yolo_outputs.append(x)  # 保存yolo模块的信息
            layer_outputs.append(x)     # 将每个块的output都保存起来
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))   # 只保存yolo层的output
        return yolo_outputs if targets is None else (loss, yolo_outputs)  # yolo层如果在训练时其output为其六个loss值；如不在训练则为其pred_box、conf、cls

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75  # 停止load

        ptr = 0  # 指针
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"] == '1':  # 因为剪枝之后的cfg没有batch_normalize时=0,并非没有此行
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                    # print(ptr)
                else:
                    # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                    if "yolov3.weights" in weights_path:
                        num_b = 255
                        ptr += num_b
                        num_w = int(self.module_defs[i-1]["filters"]) * 255
                        ptr += num_w
                    else:
                        # Load conv. bias
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                        conv_layer.weight.data.copy_(conv_w)
                        ptr += num_w
                        # print(ptr)
        # 确保指针到达权重的最后一个位置
        assert ptr == len(weights)

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            # print(module)
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # print(module[0])
                # If batch norm, load bn first
                if module_def["batch_normalize"] == '1':
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def load_darknet_weights_resprune(self, weights_path, respruneidx):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75  # 停止load

        ptr = 0  # 指针
        convidx = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                convidx += 1
                if convidx in respruneidx:
                    conv_layer = module[0]
                    if module_def["batch_normalize"] == '1':  # 因为剪枝之后的cfg没有batch_normalize时=0,并非没有此行
                        print('prune:', conv_layer, ptr)
                        # Load BN bias, weights, running mean and running variance
                        bn_layer = module[1]
                        num_b = bn_layer.bias.numel()  # Number of biases
                        # Bias
                        ptr += num_b
                        # Weight
                        ptr += num_b
                        # Running Mean
                        ptr += num_b
                        # Running Var
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        ptr += num_w
                        # print(ptr)
                    else:
                        # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                        if "yolov3.weights" in weights_path:
                            num_b = 255
                            ptr += num_b
                            num_w = int(self.module_defs[i-1]["filters"]) * 255
                            ptr += num_w
                        else:
                            # Load conv. bias
                            num_b = conv_layer.bias.numel()
                            ptr += num_b
                            # Load conv. weights
                            num_w = conv_layer.weight.numel()
                            ptr += num_w
                            # print(ptr)
                else:
                    conv_layer = module[0]
                    if module_def["batch_normalize"] == '1':  # 因为剪枝之后的cfg没有batch_normalize时=0,并非没有此行
                        # Load BN bias, weights, running mean and running variance
                        bn_layer = module[1]
                        num_b = bn_layer.bias.numel()  # Number of biases
                        # Bias
                        bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                        bn_layer.bias.data.copy_(bn_b)
                        ptr += num_b
                        # Weight
                        bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                        bn_layer.weight.data.copy_(bn_w)
                        ptr += num_b
                        # Running Mean
                        bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_rm)
                        ptr += num_b
                        # Running Var
                        bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_rv)
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                        conv_layer.weight.data.copy_(conv_w)
                        ptr += num_w
                        # print(ptr)
                    else:
                        # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                        if "yolov3.weights" in weights_path:
                            num_b = 255
                            ptr += num_b
                            num_w = int(self.module_defs[i-1]["filters"]) * 255
                            ptr += num_w
                        else:
                            # Load conv. bias
                            num_b = conv_layer.bias.numel()
                            conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                            conv_layer.bias.data.copy_(conv_b)
                            ptr += num_b
                            # Load conv. weights
                            num_w = conv_layer.weight.numel()
                            conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                            conv_layer.weight.data.copy_(conv_w)
                            ptr += num_w
                            # print(ptr)
        # 确保指针到达权重的最后一个位置
        assert ptr == len(weights)