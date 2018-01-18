# -*- coding:utf-8 -*-

#定义前向传播的过程以及神经网络中的参数

import tensorflow as tf

INPUT_NODE = 784  #输入层的节点数，图片为28*28，为图片的像素
OUTPUT_NODE = 10   #输出层的节点数，等于类别的数目，需要区分0-9，所以为10类
#配置神经网络的参数
LAYER1_NODE = 500 #隐藏层的节点数，此神经网络只有一层隐藏层

#通过tf.get_variable来获取变量，在训练网络时会创建这些变量；在测试时会通过保存的模型加载这些变量
#可以在变量加载的时候对变量重命名，可以通过同样的名字在训练时使用变量自身，而在测试时使用变量的最终值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

    #当给出正则化函数时，将当前变量的正则化损失加入losses的集合中
    #add_to_collection函数讲一个张量加入一个集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    #声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope("layer1"):
        #训练为第一次调用，所以reuse不用设置为True
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)

    #声明第二层神经网络
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights)+biases

    #返回最后前向传播结果
    return layer2