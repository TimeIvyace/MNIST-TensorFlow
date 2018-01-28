# -*- coding:utf-8 -*-

#定义前向传播的过程以及神经网络中的参数

import tensorflow as tf

INPUT_NODE = 784  #输入层的节点数，图片为28*28，为图片的像素
OUTPUT_NODE = 10   #输出层的节点数，等于类别的数目，需要区分0-9，所以为10类

IMAGE_SIZE = 28
NUM_CHANNELS = 1 #处理的图像深度
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512


# 定义卷积神经网络的前向传播过程，train用于区分训练过程和测试过程
# 增加了dropout方法，进一步提高模型可靠性防止过拟合，dropout只在训练时使用
def inference(input_tensor, train, regularizer):
    # 声明第一层卷积层的变量并实现前向传播，通过不同的命名空间来隔离不同层的变量
    # 让每一层中的变量命名只需要考虑在当前层的作用，不需要考虑重名
    # 因为卷积层输入为28*28*1，使用全0填充
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程，选用最大池化，过滤器边长为2，步长2，全0填充
    # 池化层的输入是上层卷积层的输出，也就是28*28*32的矩阵，输出为14*14*32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第三层卷积层前向传播过程，这一层输入为14*14*32，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为64的过滤器，步长为1，全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weight, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程，与第二层结构一样，输入为14*14*64，输出为7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将第四层池化层的输出转化为第五层全连接的输入格式，需要将7*7*64拉成一个向量
    # pool2.get_shape可以得到第四层输出矩阵的维度
    # 每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数
    pool_shape = pool2.get_shape().as_list()
    # 计算将矩阵拉直成向量之后的长度，这个长度是矩阵长度以及深度的乘积
    # pool_shape[0]为一个batch中数据的个数,[1][2]分别为长宽，[3]为深度
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape函数将第四层的输出变成一个batch的向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层全连接层的前向传播过程。输入为拉直的向量，长度3136，输出512的向量
    # 引入dropout，在训练时会将部分节点的输出改为0，一般只在全连接层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weight = tf.get_variable("weight", [nodes, FC_SIZE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weight))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5) # 0.5为每个元素被保存下来的概率

    # 声明第六层全连接层的前向传播过程，这一层的输入为长度是512的向量，输出为10的一维向量
    # 结果需要通过softmax层
    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable("weight", [FC_SIZE, OUTPUT_NODE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weight))
        fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weight) + fc2_biases

    #返回第六层的输出
    return logit