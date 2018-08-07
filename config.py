# -*- coding: utf-8 -*-


#采用k-means聚类方法对训练集中的groud truth做了聚类分析，得到的先验框，只有宽度个高度，没有中心点坐标，因为也没有必要
anchors = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

def read_coco_labels():   #读取coco数据集类别
    f = open("./yolo2_data/coco_classes.txt")
    class_names = []
    for l in f.readlines():
        l = l.strip()     #去掉字符串左右两遍的空格
        class_names.append(l)
    return class_names

class_names = read_coco_labels()   #返回类别
