import tensorflow as tf
import numpy as np

#the function of "tf.pad()" , maybe you can turn to the document
# array=[[1, 2, 3], [4, 5, 6]]
# t=[[1,2],[2,3]]
# x=tf.pad(array,t)
# sess=tf.Session()
# print(sess.run(x))

# anchors = [[0.57273, 0.677385],
#            [1.87446, 2.06253],
#            [3.33843, 5.47434],
#            [7.88282, 3.52778],
#            [9.77052, 9.16828]]
#
# anchors = tf.constant(anchors, dtype=tf.float32)  # 将anchors转变成tf格式的常量列表
# sess=tf.Session()
# print(sess.run(anchors))

# array=[[[1,2,3,4],
#   [5,6,7,8]],
#
#  [[9,10,11,12],
#   [13,14,15,16]]]
# array=tf.constant(array,dtype=tf.float32)
# print("--------------------------------------------------")
# #RESHAPE一下
# after_reshape=tf.reshape(array,[2,-1,2,2,2])
# sess=tf.Session()
# print(sess.run(array))
# print(sess.run(after_reshape))


# # 构建特征图每个cell的左上角的xy坐标
# sess=tf.Session()
# height_index = tf.range(3, dtype=tf.float32)  # range(0,13)
# width_index = tf.range(3, dtype=tf.float32)  # range(0,13)
# print("begin")
# print(sess.run(height_index))
# print("分割线-----------------")
# print(sess.run(width_index))
# # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
# x_cell, y_cell = tf.meshgrid(height_index, width_index)
# print("middle")
# print(sess.run(x_cell))
# print("分割线------------------")
# print(sess.run(y_cell))
#
# x_cell = tf.reshape(x_cell, [1, -1, 1])  # 和上面[H*W,num_anchors,num_class+5]对应
# y_cell = tf.reshape(y_cell, [1, -1, 1])
# sess=tf.Session()
# print("end")
# print(sess.run(x_cell))
# print("分割线------------------")
# print(sess.run(y_cell))

# box1=np.transpose([[1,2,3,4],[5,6,7,8]])
# print(box1)
#
# index=np.where([True,False,True])
# print(index)

# index=np.reshape([13, 13], [1, 1, 1, 2])
# print(index)

# test=tf.maximum([1,2], [2,1])
# a=tf.pow(2,2)
# sess=tf.Session()
# print(sess.run(test))
# print(sess.run((a)))

x = tf.constant([[[1],[2], [3]], [[4], [5], [6]]])

sess=tf.Session()

result=tf.equal([1,2,3], [3])
tf.cast(result, tf.float32)

print(sess.run(tf.cast(result, tf.float32)))