# MNIST-TensorFlow
Using Neural Networks to deal with MNIST data
四个压缩包中分别是训练数据及其标签，测试数据及其标签。
train.py为TensorFlow程序，程序中有自动调用数据包的代码，只要压缩包放在相应的文件夹下。
若相应的文件夹下无数据调用，则此代码会自动下载MNIST数据至此文件夹下。
train.py具体代码理解可见：http://blog.csdn.net/gaoyueace/article/details/79060443
train_improved1.py文件中使用tf.get_variable和tf.variable_scope函数进行优化，代码可读性更高。
train_improved1.py具体代码理解可见：http://blog.csdn.net/gaoyueace/article/details/79079068
为了让神经网络训练结果可以复用，需要将训练的到的神经网络模型持久化。
train_improved2文件夹中一共三个py程序，将训练的到的神经网络模型持久化，使用时同时运行mnist_train.py(用于训练)和mnist_eval.py(用于测试)即可。
train_improved2文件夹中代码为MNIST处理终极版，具体理解可见：http://blog.csdn.net/gaoyueace/article/details/79102149
注意：使用代码需要修改MNIST数据集存储位置以及神经网络存储位置。
