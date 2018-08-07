# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

#这些参数分别是predictions:预测值，targets:真实值，anchors:先验框，scales:loss不同部分的前面系数，num_classes:类的数量，output_size=(13,13)：特征图大小
def compute_loss(predictions,targets,anchors,scales,num_classes=20,output_size=(13,13)):  #损失函数
    W,H = output_size # H表示高度，W表示宽度，都为13，特征图大小
    C = num_classes   #num_classes为20，说明用的是pascal_voc数据集
    B = len(anchors)    #先验框的数量
    anchors = tf.constant(anchors,dtype=tf.float32)  #将先验框变成tensorflow张量
    anchors = tf.reshape(anchors,[1,1,B,2])  #存放输入的anchors的wh，论文中B=5

    # 【1】ground truth：期望值、真实值
    sprob,sconf,snoob,scoor = scales #loss不同部分的前面系数
    _coords = targets["coords"]  # ground truth [-1, H*W, B, 4]，真实坐标xywh
    _probs = targets["probs"]  # class probability [-1, H*W, B, C] ，类别概率——one hot形式，C维
    _confs = targets["confs"]  # 1 for object, 0 for background, [-1, H*W, B]，置信度，每个边界框一个
    # ground truth计算IOU-->_up_left, _down_right
    _wh = tf.pow(_coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
    _centers = _coords[:, :, :, 0:2]
    _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
    # ground truth汇总
    truths = tf.concat([_coords, tf.expand_dims(_confs, -1), _probs], 3)

    # 【2】decode the net prediction：预测值、网络输出值
    predictions = tf.reshape(predictions,[-1,H,W,B,(5+C)])  #这四个维度分别是batch_size, 特征图高度，特征图宽度，每个cell的框的数量，5+num_classes
    # t_x, t_y, t_w, t_h
    coords = tf.reshape(predictions[:,:,:,:,0:4],[-1,H*W,B,4]) # reshape成[batch_size, H*W, 5, 4]
    coords_xy = tf.nn.sigmoid(coords[:,:,:,0:2])  # 0-1，xy是相对于cell左上角的偏移量, 计算得出的是偏移量
    coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4])*anchors/np.reshape([W,H],[1,1,1,2])) # 0-1，除以特征图的尺寸13，解码成相对于整张图片的wh
    coords = tf.concat([coords_xy,coords_wh],axis=3) # [batch_size, H*W, B, 4]
    # 置信度
    confs = tf.nn.sigmoid(predictions[:,:,:,:,4])    #经过一个sigmoid函数，使其落在[0-1]之间
    confs = tf.reshape(confs,[-1,H*W,B,1]) # 每个边界框一个置信度，每个cell有B个边界框，reshape成[batch_size, H*W, 5, 1]
    #  类别概率
    probs = tf.nn.softmax(predictions[:,:,:,:,5:]) # 网络最后输出是"得分"，通过softmax变成概率
    probs = tf.reshape(probs,[-1,H*W,B,C])   #reshape成[batch_size, H*W, B, C]
    # prediction汇总
    preds = tf.concat([coords,confs,probs],axis=3) #  reshape成[-1, H*W, B, (4+1+C)]
    # prediction计算IOU-->up_left, down_right
    wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])  #
    areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    up_left, down_right = centers - (wh * 0.5), centers + (wh * 0.5)  #左上角，右下角


    #【3】计算ground truth和anchor的IOU：
    # 计算IOU只考虑形状，先将anchor与ground truth的中心点都偏移到同一位置（cell左上角），然后计算出对应的IOU值。
    # ①IOU值最大的那个anchor与ground truth匹配，对应的预测框用来预测这个ground truth：计算xywh、置信度c(目标值为1)、类别概率p误差。
    # ②IOU小于某阈值的anchor对应的预测框：只计算置信度c(目标值为0)误差。
    # ③剩下IOU大于某阈值但不是max的anchor对应的预测框：丢弃，不计算任何误差。
    inter_upleft = tf.maximum(up_left, _up_left)   #得到ground truth和anchor交互部分IOU左上角的坐标
    inter_downright = tf.minimum(down_right, _down_right)  #得到ground truth和anchor交互部分右下角的坐标
    inter_wh = tf.maximum(inter_downright - inter_upleft, 0.0)
    intersects = inter_wh[:, :, :, 0] * inter_wh[:, :, :, 1]   #得到
    ious = tf.truediv(intersects, areas + _areas - intersects)   #得到IOU， 预测值与真实值进行对比了

    best_iou_mask = tf.equal(ious, tf.reduce_max(ious, axis=2, keep_dims=True))   #每一个cell找到一个最大IOU，返回True或者False，在第三个维度上，第三个维度即是box数量=5
    best_iou_mask = tf.cast(best_iou_mask, tf.float32)   #将True或者False转换成1或者0， False->0, True->1
    mask = best_iou_mask * _confs  # [-1, H*W, B]  conf为真实值
    mask = tf.expand_dims(mask, -1)  # [-1, H*W, B, 1]   # 真实有目标的区域为1，没有的为0

    # 【4】计算各项损失所占的比例权重weight
    confs_w = snoob * (1 - mask) + sconf * mask   #分别计算出没有目标的置信度和有目标的置信度， [-1,H*W,B,1]
    coords_w = scoor * mask   #得到有目标区域的分数
    probs_w = sprob * mask    #类别置信度
    weights = tf.concat([coords_w, confs_w, probs_w], axis=3)

    # 【5】计算loss：ground truth汇总和prediction汇总均方差损失函数，再乘以相应的比例权重
    loss = tf.pow(preds - truths, 2) * weights
    loss = tf.reduce_sum(loss, axis=[1, 2, 3])
    loss = 0.5 * tf.reduce_mean(loss)

    return loss
