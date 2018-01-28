# -*- coding:utf-8 -*-

#定义神经网络的训练过程

import os  #os模块是对操作系统进行调用的接口
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference

#配置神经网络的参数
BATCH_SIZE = 100 #一个训练batch中的训练数据个数，数字越小，越接近随机梯度下降，越大越接近梯度下降
LEARNING_RATE_BASE = 0.01 #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #描述网络复杂度的正则化向在损失函数中的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率
#模型保存的路径和文件名
MODEL_SAVE_PATH = "/tensorflow_google/mnist"
MODEL_NAME = "mnist.ckpt"

def train(mnist):
    #定义输入输出，卷积神经网络的输入层为一个三维矩阵
    #第一维为一个batch中样例的个数，第二维和第三维表示图片的尺寸，第四维表示图片的深度
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    #正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #直接使用mnist-inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, 1, regularizer)

    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均，需要被训练的参数,variable_averages返回的就是GraphKeys.TRAINABLE_VARIABLES中的元素
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵，使用了sparse_softmax_cross_entropy_with_logits，当问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。
    # 这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案,argmax返回最大值的位置
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    # LEARNING_RATE_BASE为基础学习率，global_step为当前迭代的次数
    # mnist.train.num_examples/BATCH_SIZE为完整的过完所有的训练数据需要的迭代次数
    # LEARNING_RATE_DECAY为学习率衰减速度

    # 使用GradientDescentOptimizer优化算法优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 在训练神经网络的时候，每过一遍数据都要通过反向传播来更新参数以及其滑动平均值
    # 为了一次完成多个操作，可以通过tf.control_dependencies和tf.group两种机制来实现
    # train_op = tf.group(train_step, variable_averages_op)  #和下面代码功能一样
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #在训练过程中不再测试网络在验证数据上的表现，验证和测试的过程将会有一个独立的程序
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            #将输入的训练数据调整为一个四维矩阵
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})
            #每1000轮保存一次模型
            if i%1000==0:
                #输出当前的训练情况，只输出了网络在当前训练batch上的损失函数大小
                print("After %d training step(s), loss on training batch is %g," %(step, loss_value))
                #保存当前的网络，给出了global_step参数，这样可以让每个保存网络的文件名末尾加上训练的轮数
                #例如mnist.ckpt-1000
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/tensorflow_google", one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()