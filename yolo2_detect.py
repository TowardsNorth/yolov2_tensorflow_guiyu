import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
from model_darknet19 import darknet
from decode import decode
from utils import preprocess_image, postprocess, draw_detection, generate_colors
from config import anchors, class_names


model_path = os.path.join('yolo2_model','yolo2_coco.ckpt')    #加载模型路径

image_name ='car3.jpg'
image_file = os.path.join('images',image_name)   #images/car3.jpg
image_detection = os.path.join('images',"dect_car3.jpg") #images/dect_cat3.jpg

image = cv2.imread(image_file)  #read the image, images/car3.jpg
image_shape = image.shape[:2]

input_size = (416,416)

image_cp = preprocess_image(image)  #图像预处理，resize image, normalization归一化， 增加一个在第0的维度--batch_size
tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3])  #定义placeholder
model_output = darknet(tf_image)  #网络的输出

output_sizes = input_size[0]//32, input_size[1]//32 # 特征图尺寸是图片下采样32倍

#这个函数返回框的坐标（左上角-右下角），目标置信度，类别置信度
output_decoded = decode(model_output=model_output,output_sizes=output_sizes, num_class=len(class_names),anchors=anchors)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   #初始化tensorflow全局变量
    saver = tf.train.Saver()
    saver.restore(sess, model_path)  #把模型加载到当前session中
    bboxes, obj_probs, class_probs = sess.run(output_decoded, feed_dict={tf_image: image_cp})  #这个函数返回框的坐标，目标置信度，类别置信度


bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)   #得到候选框之后的处理，先留下阈值大于0.5的框，然后再放入非极大值抑制中去
colors = generate_colors(class_names)
img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names, colors)  #得到图片

cv2.imwrite(image_detection, img_detection)
cv2.imshow("detection_results", img_detection)  #显示图片
cv2.waitKey(0)  #等待按任意键退出



