# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np


# 激活函数
def leaky_relu(x):    #leaky relu激活函数，leaky_relu激活函数一般用在比较深层次神经网络中
    return tf.maximum(0.1*x,x)
    #return tf.nn.leaky_relu(x,alpha=0.1,name='leaky_relu') # 或者tf.maximum(0.1*x,x)

    
# Conv+BN：yolo2中每个卷积层后面都有一个BN层， batch normalization正是yolov2比yolov1多的一个东西，可以提升mAP大约2%
def conv2d(x,filters_num,filters_size,pad_size=0,stride=1,batch_normalize=True,activation=leaky_relu,use_bias=False,name='conv2d'):
    # padding，注意: 不用padding="SAME",否则可能会导致坐标计算错误，用自己定义的填充方式
    if pad_size > 0:
        x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])  #这里使用tensorflow中的pad进行填充,主要填充第2和第3维度，填充的目的是使得经过卷积运算之后，特征图的大小不会发生变化
  
    out = tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,padding='VALID',activation=None,use_bias=use_bias,name=name)
    # BN应该在卷积层conv和激活函数activation之间,(后面有BN层的conv就不用偏置bias，并激活函数activation在后)
    if batch_normalize:      #卷积层的输出，先经过batch_normalization
        out = tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
    if activation:   #经过batch_normalization处理之后的网络输出输入到激活函数
        out = activation(out)
    return out   #返回网络的输出

# max_pool
def maxpool(x,size=2,stride=2,name='maxpool'):      #maxpool，最大池化层
    return tf.layers.max_pooling2d(x,pool_size=size,strides=stride)

# reorg layer(带passthrough的重组层)，主要是利用到Fine-Grained Feature（细粒度特征用于检测微小物体）
def reorg(x,stride):
    return tf.space_to_depth(x,block_size=stride)   #返回一个与input具有相同的类型的Tensor
    # return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')

# Darknet19 
# 默认是coco数据集，最后一层维度是anchor_num*(class_num+5)=5*(80+5)=425,注意与voc数据集的区别
def darknet(images,n_last_channels=425):     #Darknet19网络，假设这里图片的输入大小为224*224，下面的所有注释都是基于这个宽度和高度的假设,主要是为了和论文对应
    net = conv2d(images, filters_num=32, filters_size=3, pad_size=1, name='conv1') #卷积层，卷积核数量32，大小为3*3，padding=1, 默认步长为1
    net = maxpool(net, size=2, stride=2, name='pool1')    #maxpooling, 减少特征图的维度一半，为112*112,因为siez=2*2,步长为2

    net = conv2d(net, 64, 3, 1, name='conv2')  #卷积层，卷积核数量为64，大小为3*3，padding=1,默认步长为1
    net = maxpool(net, 2, 2, name='pool2')     #maxpooling，变成56*56

    net = conv2d(net, 128, 3, 1, name='conv3_1')  #卷积层，卷积核数量为128，大小为3*3，padding=1,默认步长为1
    net = conv2d(net, 64, 1, 0, name='conv3_2')   #卷积层，卷积核数量为64，大小为1*1，padding=0,默认步长为1
    net = conv2d(net, 128, 3, 1, name='conv3_3')  #卷积层，卷积核数量为128，大小为3*3，padding为1，默认步长为1
    net = maxpool(net, 2, 2, name='pool3')     #maxpooling,变成28*28

    net = conv2d(net, 256, 3, 1, name='conv4_1')   #卷积层，卷积核数量为256，大小为3*3，padding=1,默认步长为1
    net = conv2d(net, 128, 1, 0, name='conv4_2')   #卷积层，卷积核数量为128，大小为1*1，padding=0，默认步长为1
    net = conv2d(net, 256, 3, 1, name='conv4_3')   #卷积层，卷积核数量为256，大小为3*3，padding=1,默认步长为1
    net = maxpool(net, 2, 2, name='pool4')       #maxpooling,变成14*14

    net = conv2d(net, 512, 3, 1, name='conv5_1')  #卷积层，卷积核数量为512，大小为3*3，padding=1,默认步长为1
    net = conv2d(net, 256, 1, 0,name='conv5_2')   #卷积层，卷积核数量为256，大小为1*1，padding=0,默认步长为1
    net = conv2d(net,512, 3, 1, name='conv5_3')   #卷积层，卷积核数量为512，大小为3*3，padding=1,默认步长为1
    net = conv2d(net, 256, 1, 0, name='conv5_4')  #卷积层，卷积核数量为256，大小为1*1，padding=0,默认步长为1
    net = conv2d(net, 512, 3, 1, name='conv5_5')   #卷积层，卷积核数量为512，大小为1*1,padding=1,默认步长为1
    
    # 存储这一层特征图，以便后面passthrough层
    shortcut = net      #大小为14*14
    net = maxpool(net, 2, 2, name='pool5')  #maxpooling，变成7*7

    net = conv2d(net, 1024, 3, 1, name='conv6_1')  #卷积层，卷积核数量为1024,大小为3*3，padding=1,默认步长为1
    net = conv2d(net, 512, 1, 0, name='conv6_2')   #卷积层，卷积核数量为512，大小为1*1，padding=0,默认步长为1
    net = conv2d(net, 1024, 3, 1, name='conv6_3')  #卷积层，卷积核数量为1024，大小为3*3，padding=1，默认步长为1
    net = conv2d(net, 512, 1, 0, name='conv6_4')  #卷积层，卷积核数量为512，大小为1*1，padding=0,默认步长为1
    net = conv2d(net, 1024, 3, 1, name='conv6_5')  #卷积层，卷积核数量为1024，大小为3*3，padding=1,默认步长为1

    #具体这个可以参考： https://blog.csdn.net/hrsstudy/article/details/70767950     Training for classification 和 Training for detection
    # 训练检测网络时去掉了分类网络的网络最后一个卷积层，在后面增加了三个卷积核尺寸为3 * 3，卷积核数量为1024的卷积层，并在这三个卷积层的最后一层后面跟一个卷积核尺寸为1 * 1
    # 的卷积层，卷积核数量是（B * （5 + C））。
    # 对于VOC数据集，卷积层输入图像尺寸为416 * 416
    # 时最终输出是13 * 13
    # 个栅格，每个栅格预测5种boxes大小，每个box包含5个坐标值和20个条件类别概率，所以输出维度是13 * 13 * 5 * （5 + 20）= 13 * 13 * 125。
    #
    # 检测网络加入了passthrough
    # layer，从最后一个输出为26 * 26 * 512
    # 的卷积层连接到新加入的三个卷积核尺寸为3 * 3
    # 的卷积层的第二层，使模型有了细粒度特征。


    #下面这部分主要是training for detection
    net = conv2d(net, 1024, 3, 1, name='conv7_1')  #卷积层，卷积核数量为1024，大小为3*3,padding=1,默认步长为1
    net = conv2d(net, 1024, 3, 1, name='conv7_2')  #卷积层，卷积核数量为1024，大小为3*3，padding=1,默认步长为1，大小为1024*7*7

    #关于这部分细粒度的特征的解释，可以参考：https://blog.csdn.net/hai_xiao_tian/article/details/80472419
    # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
    # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图，可能是输入图片刚开始不是224，而是448，知道就好了,YOLOv2的输入图片大小为 416*416
    shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')  #卷积层，卷积核数量为64，大小为1*1，padding=0,默认步长为1，变成26*26*64
    shortcut = reorg(shortcut, 2)  # passthrough, 进行Fine-Grained Features，得到13*13*256
    #连接之后，变成13*13*（1024+256）
    net = tf.concat([shortcut, net], axis=-1)  #channel整合到一起，concatenated with the original features，passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上，
    net = conv2d(net, 1024, 3, 1, name='conv8')  #卷积层，卷积核数量为1024，大小为3*3，padding=1, 在连接的特征图的基础上做卷积进行预测。变成13*13*1024

    #detection layer: 最后用一个1*1卷积去调整channel，该层没有BN层和激活函数，变成: S*S*(B*(5+C))，在这里为：13*13*425
    output = conv2d(net, filters_num=n_last_channels, filters_size=1, batch_normalize=False, activation=None, use_bias=True, name='conv_dec')

    return output   #返回网络的输出



if __name__ == '__main__':
    x = tf.random_normal([1, 416, 416, 3])
    model_output = darknet(x)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 必须先restore模型才能打印shape;导入模型时，上面每层网络的name不能修改，否则找不到
        saver.restore(sess, "./yolo2_model/yolo2_coco.ckpt")
        print(sess.run(model_output).shape) # (1,13,13,425)
