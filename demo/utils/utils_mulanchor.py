from __future__ import division
import tqdm
import torch
import numpy as np
import random
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:, -1])
        except:
            continue
        # WE will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):  # 初始化权重
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        # torch.nn.init.normal_(m.weight.data, 0.0, math.sqrt(2. / 1.0001 * n))
        torch.nn.init.kaiming_uniform_(m.weight.data, 0.01)
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        # torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):  # 各个类别的ap值
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)  # argsort返回的是数组值从小到大的索引值（加负号为从大到小）
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # get_batch_statistics函数输出的有效值 # 使得数组中元素按置信度从大到小排列
    # 这些值是按照置信度降低排列统计的，可以认为是取不同的置信度阈值（或者rank值）得到的 相当于把每一类的位置作为不同的阈值
    # Find unique classes
    unique_classes = np.unique(target_cls)  # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表


    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []  # 所有类的ap值 精度 召回率存储
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):  # 分别计算每一类的ap值
        i = pred_cls == c  # 返回 true or false 的数组 (可作 int 0或1) 例如[false false false false]亦可作[0 0 0 0]
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects(c类)

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()  # cumsum: axis=0，按照行累加。 axis=1，按照列累加 (1 - tp[i])：使1变为0，使0变为1  （下面置信度阈值判断下tp为0的即fp变为1）
            tpc = (tp[i]).cumsum()  # i即代表pred_cls中所有是c类的id号，tp[i]可输出此处（是此类的）的box其对应的tp值(tp值即为虽然判别为该类了，但是否可信还要由置信度阈值决定)
            # 输出的是逐渐累计增加的tp和fp，以便绘制recall和precision的图像
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)  # tp/(tp+fn)  多个recall值
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)  # tp/(tp+fp)  多个presision值
            p.append(precision_curve[-1])  # 加上其最后一个值

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))  # ap值是针对某一分类的 ap是一个数组

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):  # 输入的是recall和precision的curve
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope # 计算出precision的各个断点(折线点)
    for i in range(mpre.size - 1, 0, -1):  # range(start, stop[, step])
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # maximum：X和Y逐位进行比较,选择最大值.最少接受两个参数 只留下mpre的峰值 eg:[1,2,8,5,7]变为[8 8 8 7 7]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]  # recall前后两个值不一样的点 size:(array([0, 1, 2, 3]),) 故[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])   # 峰值折线点下面积的和(相当于recall值逐个与上个相减再乘以峰值的pre了)
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):  # 输出有效的预测框 设置iou的阈值（定值，默认为0.5）
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):  # output中的各个

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]  #从output中提取三类信息
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]  # 提取同一张图片中的真值的label x y w h
        target_labels = annotations[:, 0] if len(annotations) else []  # 提取真值label
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]  # 提取真值框

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):  # 所有真值框的targets都已被检测到,则break
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)  # pred_boxes中与target_boxes iou最大的那一个
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1  # pred中的tp[i]值设为1
                    detected_boxes += [box_index]  # 大于iou阈值的box_index列入detected_boxes中
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2  # 中心点减去宽和高的二分之一 即左上角和右下角的x坐标
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )  # clamp: 将input中的元素限制在[min,max]范围内并返回一个Tensor  torch.clamp(input,min,max,out=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):  # 阈值限制 非极大值抑制
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor  # 标志位类型
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)     # num_samples(batch_size), num_anchor * (3*(class+5)), grid?
    nA = pred_boxes.size(1)     # num_anchors
    nC = pred_cls.size(-1)      # 维度:num_samples, num_anchors, grid_size_w, grid_size_h, num_classes  取最后一维关于classes的预测值

    nH = pred_boxes.size(2)
    nW = pred_boxes.size(3)

    obj_mask = ByteTensor(nB, nA, nH, nW).fill_(0)  # 对应yolo forward中x y的维度 高*宽
    noobj_mask = ByteTensor(nB, nA, nH, nW).fill_(1)
    class_mask = FloatTensor(nB, nA, nH, nW).fill_(0)
    iou_scores = FloatTensor(nB, nA, nH, nW).fill_(0)
    tx = FloatTensor(nB, nA, nH, nW).fill_(0)
    ty = FloatTensor(nB, nA, nH, nW).fill_(0)
    tw = FloatTensor(nB, nA, nH, nW).fill_(0)
    th = FloatTensor(nB, nA, nH, nW).fill_(0)
    tcls = FloatTensor(nB, nA, nH, nW, nC).fill_(0)  # nC的值可以取到所有label索引的值

    # Convert to position relative to box
    target_boxes = torch.stack([target[:, 2] * nW, target[:, 3] * nH, target[:, 4] * nW, target[:, 5] * nH], 1)  # target_boxes与target两个变量指向同一个元素,要改都会改  # groud truth
    # xywh实际坐标(不同的yolo层放缩程度不一样)
    # 坐标信息归一化到了0~1之间,需要进行放大(乘的是feature map的宽高 实际上也是grid的个数)
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]  # 好几个target坐标框的宽和高
    gwh2 = gwh
    gxy2 = gxy
    #****************************
    # iou of targets-anchors
    t, best_n,tbox = target, [],[]
    nt = len(target)

    use_all_anchors, reject = True, True
    if nt:
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # 每组anchor的iou按行堆叠

        if use_all_anchors:
            na = len(anchors)  # number of anchors
            best_n = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)  # bestn的索引是所有anchor索引 每一列的所有值都被返回
            # 假设na=3,nt=5,则best_n:tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
            t = t.repeat([na, 1])  # 将target框重复anchor次(按行堆叠) 因为每个都要参与输出
            gwh = gwh.repeat([na, 1])  # 同上
            gxy = gxy.repeat([na, 1])  # 同上
        else:  # use best anchor only
            best_ious, best_n = ious.max(0)  # best iou and anchor  # 最好的iou值和对应的anchor索引  # max(0)返回该矩阵中每一列的最大值(每张图)  max(1)返回该矩阵中每一行的最大值

        # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
        if reject:
            j = ious.view(-1) > 0.225  # iou threshold hyperparameter
            # mask,大于阈值的为1,小于阈值的为0
            t, best_n, gwh,gxy = t[j], best_n[j], gwh[j],gxy[j]  # mask: 每个有obj的grid和其对应的anchor与target在mask上是一一对应的,所以当一个有obj的grid对应多个anchor时,应将grid与target索引都复制多份
            if len(best_n) == 0:
                best_ious, best_n = ious.max(0)  # best iou and anchor
                gwh = gwh2
                gxy = gxy2
                t = target
            # best_n = best_n[j]
    tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)  # 实际上是过滤之后的重复几次的target_boxes
    '''
    # [tensor([[15.5958, 11.2643,  4.5164,  2.9755],
    #         [ 8.4154, 13.4060, 11.2558,  4.8392],
    #         [15.5958, 11.2643,  4.5164,  2.9755],
    #         [12.3144, 13.4060,  3.4579,  4.8392],
    #         [ 8.4154, 13.4060, 11.2558,  4.8392],
    #         [15.5958, 11.2643,  4.5164,  2.9755],
    #         [12.3144, 13.4060,  3.4579,  4.8392]])]
    '''
    #****************************

   #  # Get anchors with best iou
   #  ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])    # anchors*target  anchors与ground truth的iou  ious.shape:[anchor数(3), 一个batch内的target总数]  # 只考虑wh的iou
   # # anchors*target  anchors与ground truth的iou  ious.shape:[anchor数(3), 一个batch内的target总数]  # 只考虑wh的iou
   #  best_ious, best_n = ious.max(0)  # 最好的iou值和对应的anchor索引  # max(0)返回该矩阵中每一列的最大值(每张图)  max(1)返回该矩阵中每一行的最大值
    #*****************************
    # anchor框的中心点最开始时都在原点
    # 确定用来预测的anchor 用来生成预测框 从而生成loss 其他只生成conf noobj loss
    # Separate target values
    b, target_labels = t[:, :2].long().t()  # target = [idx, labels, x, y, w, h] 第一维取所有,第二维取前2
    gx, gy = gxy.t()  # 转置
    # print(gx.floor(), gy.floor())
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # long是变成长整型  先列索引后行索引  特征图左上角坐标
    # print(gi, gj)
    # print(gx, gy, gw, gh, gi, gj)
    """
    """
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1  # 行索引列索引 # 先筛选anchor()只有anchor索引有变化: [图片idx 符合iou阈值的anchor索引 grid grid (先行索引后列索引)] 设为obj = 1 obj_mask为用于预测的anchor
    noobj_mask[b, best_n, gj, gi] = 0  # 对应设no_obj = 0  这里只考虑了负责预测的anchor的noobj loss
    # 这里的gi gj由target的x y转换而来,指有target的特定的grid
    # print(gj, gi)
    # 之后在最佳anchor的基础上继续进行预测
    if not use_all_anchors:
    # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):  # 图片索引和对应的anchors_iou
            # print(b[i], gj[i], gi[i])
            # print(i)
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0  # 一个target中大于iou阈值的anchors索引的 no_obj = 0  其他的本来就被设置为1, 设为1的需要计算no_obj置信度误差,设为0的说明此处的anchor框要么用于生成预测框,即使没有用来生成预测框,也不记入置信度误差,其预测结果被忽略.
            # 此处考虑所有anchor的no obj loss
    # 以上用来设置mask以计算对应位置的loss
    # 以下用来计算真值
    # Coordinates
    # 用于预测的anchor,逐个筛选其grid索引
    tx[b, best_n, gj, gi] = gx - gx.floor()  # gx减去gx取整,即groud truth中的偏置值
    ty[b, best_n, gj, gi] = gy - gy.floor()  # gy减去gy取整,即groud truth中的偏置值
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)  # 同理也是groud truth中的偏置值
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # 为什么要除以anchor的宽和高:  tw和th的公式yolov3和faster-rcnn系列是一样的，是物体所在边框的长宽和anchor box长宽之间的比率 再缩放到对数空间
    # 此处的anchors是scaled_anchors: self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])  # anchor的尺度除以每一个grid的宽高 维度为[ self.num_anchors,2] 即每一个anchor占据了几个grid 即按输出特征图大小缩放后的anchor
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1  # 对应索引位置的对应target label 的 class label = 1 groud truth # 用于计算类别概率
    # Compute label correctness and iou at best anchor
    # print(pred_cls.shape) [num_samples, num_anchors, grid_size_w, grid_size_h, num_classes]
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()  # argmax(-1):沿-1轴返回最大值索引  即class概率最大的那一个的索引正好等于label的情况 索引对应的class_masks为1 即预测出了正确的class的索引为1
    # print(pred_boxes.shape) [num_samples, num_anchors, grid_size_w, grid_size_h, 4(xywh)]
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], tbox[0], x1y1x2y2=False)  # 每个box的iou得分为iou值  考虑xywh的iou
    tconf = obj_mask.float()  # 置信度 groud truth的边框置信度为1
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
