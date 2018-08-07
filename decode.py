# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def decode(model_output,output_sizes=(13,13),num_class=80,anchors=None):
    '''
     model_output:darknet19网络输出的特征图
     output_sizes:darknet19网络输出的特征图大小，默认是13*13(默认输入416*416，下采样32)
    '''
    H, W = output_sizes  # H=13, W=13
    num_anchors = len(anchors) # num_anchors=5
    anchors = tf.constant(anchors, dtype=tf.float32)  # 将anchors转变成tf格式的常量列表

    # 13*13*num_anchors*(num_class+5)，第一个维度自适应batchsize
    detection_result = tf.reshape(model_output,[-1,H*W,num_anchors,num_class+5])     #注意reshape的用法

    # darknet19网络输出转化——偏移量、置信度、类别概率
    xy_offset = tf.nn.sigmoid(detection_result[:,:,:,0:2]) # 中心坐标相对于该cell坐上角的偏移量，用sigmoid函数归一化到(0,1)
    wh_offset = tf.exp(detection_result[:,:,:,2:4]) #相对于特征图的wh比例，通过e指数解码
    obj_probs = tf.nn.sigmoid(detection_result[:,:,:,4]) # 置信度，sigmoid函数归一化到(0,1)
    class_probs = tf.nn.softmax(detection_result[:,:,:,5:]) # 网络回归的是得分，用softmax转变成概率，使其值落在(0,1)

    # 构建特征图每个cell的左上角的xy坐标
    height_index = tf.range(H,dtype=tf.float32) # range(0,13)
    width_index = tf.range(W,dtype=tf.float32) # range(0,13)
    # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
    x_cell,y_cell = tf.meshgrid(height_index,width_index)
    x_cell = tf.reshape(x_cell,[1,-1,1]) # 和上面[H*W,num_anchors,num_class+5]对应
    y_cell = tf.reshape(y_cell,[1,-1,1])

    # decode
    bbox_x = (x_cell + xy_offset[:,:,:,0]) / W
    bbox_y = (y_cell + xy_offset[:,:,:,1]) / H
    bbox_w = (anchors[:,0] * wh_offset[:,:,:,0]) / W   #先验框宽度*偏移值
    bbox_h = (anchors[:,1] * wh_offset[:,:,:,1]) / H   #先验框高度*偏移值
    # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
    bboxes = tf.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2, bbox_x+bbox_w/2, bbox_y+bbox_h/2], axis=3)  #转变成坐上-右下坐标

    return bboxes, obj_probs, class_probs   #这个返回框的坐标（左上角-右下角），目标置信度，类别概率
