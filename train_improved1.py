import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  #输入层的节点数，图片为28*28，为图片的像素
OUTPUT_NODE = 10   #输出层的节点数，等于类别的数目，需要区分0-9，所以为10类

#配置神经网络的参数
LAYER1_NODE = 500 #隐藏层的节点数，此神经网络只有一层隐藏层
BATCH_SIZE = 100 #一个训练batch中的训练数据个数，数字越小，越接近随机梯度下降，越大越接近梯度下降
LEARNING_RATE_BASE = 0.8 #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARIZATION_RATE = 0.0001 #描述网络复杂度的正则化向在损失函数中的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#给定神经网络的输入和所有参数，计算神经网络的前向传播结果，定义了一个使用ReLU的三层全连接神经网络，通过加入隐藏层实现了多层网络结构
def inference(input_tensor, avg_class, reuse=False):
    #定义第一层神经网络的变量和前向传播结果
    with tf.variable_scope("layer1", reuse=reuse):
        #根据传进来的reuse来判断是创建新变量还是使用已经创建好的
        #在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需要每次传入变量了
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        # 若没有提供滑动平均类，则直接使用参数当前的取值
        if avg_class == None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))

    #定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            layer2 = tf.matmul(layer1, weights)+biases
        else:
            layer2 = tf.matmul(layer1, avg_class.average(weights))+avg_class.average(biases)
    #返回最后的前向传播结果
    return layer2


#训练网络的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    #计算在当前参数下神经网络前向传播的结果，这里的用于计算滑动平均的类为None，所以没有使用滑动平均值
    y = inference(x, None)
    #在程序中需要使用训练好的神经网络进行推导时，可直接调用inference(new_x, variable_averages, True)

    #定义存储训练轮数的变量，这个变量不需要被训练
    global_step = tf.Variable(0, trainable=False)

    #初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    #在所有代表神经网络参数的变量上使用滑动平均，需要被训练的参数,variable_averages返回的就是GraphKeys.TRAINABLE_VARIABLES中的元素
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用了滑动平均之后的前向传播结果，滑动平均不会改变变量本身取值，会用一个影子变量来记录
    average_y = inference(x, variable_averages, True)

    #计算交叉熵，使用了sparse_softmax_cross_entropy_with_logits，当问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。
    #这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案,argmax返回最大值的位置
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    with tf.variable_scope("", reuse=True):
        weights1 = tf.get_variable("layer1/weights", [INPUT_NODE, LAYER1_NODE])
        weights2 = tf.get_variable("layer2/weights", [LAYER1_NODE, OUTPUT_NODE])

    #计算网络的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失为交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    #LEARNING_RATE_BASE为基础学习率，global_step为当前迭代的次数
    #mnist.train.num_examples/BATCH_SIZE为完整的过完所有的训练数据需要的迭代次数
    #LEARNING_RATE_DECAY为学习率衰减速度

    #使用GradientDescentOptimizer优化算法优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #在训练神经网络的时候，每过一遍数据都要通过反向传播来更新参数以及其滑动平均值
    # 为了一次完成多个操作，可以通过tf.control_dependencies和tf.group两种机制来实现
    # train_op = tf.group(train_step, variable_averages_op)  #和下面代码功能一样
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = 'train')

    #检验使用了滑动平均模型的神经网络前向传播结果是否正确
    #f.argmax(average_y, 1)计算了每一个样例的预测答案，得到的结果是一个长度为batch的一维数组
    #一维数组中的值就表示了每一个样例对应的数字识别结果
    #tf.equal判断两个张量的每一维是否相等。如果相等返回True，反之返回False
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #首先将一个布尔型的数组转换为实数，然后计算平均值
    #平均值就是网络在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #初始会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run() #参数初始化
        #准备验证数据，在神经网络的训练过程中，会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_data = {x: mnist.validation.images, y_:mnist.validation.labels}
        #准备测试数据
        test_data = {x:mnist.test.images, y_:mnist.test.labels}
        #迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i%1000==0:
                #计算滑动平均模型在验证数据上的结果，因为MNIST数据集较小，所以可以一次处理所有的验证数据
                validate_acc = sess.run(accuracy, feed_dict=validate_data)
                print("After %d training steps, validation accuracy using average model is %g"
                      %(i, validate_acc))

            # 产生训练数据batch,开始训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)  # xs为数据，ys为标签
            sess.run(train_op, feed_dict={x:xs, y_:ys})

        test_acc = sess.run(accuracy, feed_dict=test_data)
        print("After %d training steps, validation accuracy using average model is %g"
              %(TRAINING_STEPS, test_acc))

#程序主入口
def main(argv=None):
    # 声明处理MNIST数据集的类,one_hot=True将标签表示为向量形式
    mnist = input_data.read_data_sets("/Users/gaoyue/文档/Program/tensorflow_google/chapter5", one_hot=True)
    train(mnist)

#TensorFlow提供程序主入口，tf.app.run会调用上面定义的main函数
if __name__ =='__main__':
    tf.app.run()