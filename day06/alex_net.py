import tensorflow as tf

# 构建AlexNet模型
net = tf.keras.models.Sequential([
    #  卷积层, 96个卷积核，卷积核为11*11，步幅为4，激活函数relu
    tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu'),
    #  最大池化层，池化核为3*3，步幅为2
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    #  卷积层, 256个卷积核，卷积核为5*5，步幅为1，padding为same 激活函数relu
    tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    #  最大池化层，池化核为3*3，步幅为2
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    #  卷积层, 384个卷积核，卷积核为3*3，步幅为1，padding为same 激活函数relu
    tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),

    #  卷积层, 384个卷积核，卷积核为3*3，步幅为1，padding为same 激活函数relu
    tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    
    #  卷积层, 256个卷积核，卷积核为3*3，步幅为1，padding为same 激活函数relu
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    #  最大池化层，池化核为3*3，步幅为2
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    #  将多维输入一维化
    tf.keras.layers.Flatten(),
    #  全连接层，4096个神经元，激活函数relu
    tf.keras.layers.Dense(4096, activation='relu'),
    # 随机失活
    tf.keras.layers.Dropout(0.5),
    #  全连接层，4096个神经元，激活函数relu
    tf.keras.layers.Dense(4096, activation='relu'),
    # 随机失活
    tf.keras.layers.Dropout(0.5),
    #  全连接层，10个神经元，激活函数softmax
    tf.keras.layers.Dense(10, activation='softmax')
])

# x = tf.random.uniform(shape=(1, 227, 227, 1))
# y = net(x)
# print(net.summary())


import numpy as np

# 获取手写数字数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 训练集数据维度的调整， N H W C
train_images = np.reshape(train_images,(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
# 测试集数据维度的调整， N H W C
test_images = np.reshape(test_images,(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

# 获取训练数据集
def get_train(size):
    # 随机生成要抽样的样本的索引
    index = np.random.randint(0, np.shape(train_images)[0], size=size)
    # 将这些数据resize成227*227大小
    resized_images = tf.image.resize_with_pad(train_images[index], 227, 227)
    # 返回抽取的数据
    return resized_images.numpy(), train_labels[index]

# 获取测试数据集
def get_test(size):
    # 随机生成要抽样的样本的索引
    index = np.random.randint(0, np.shape(test_images)[0], size=size)
    # 将这些数据resize成227*227大小
    resized_images = tf.image.resize_with_pad(test_images[index], 227, 227)
    # 返回抽取的数据
    return resized_images.numpy(), test_labels[index]

# 获取训练样本和测试样本
train_images, train_labels = get_train(5000)  # 增加训练样本数到5000
test_images, test_labels = get_test(1000)    # 增加测试样本数到1000

print(train_images.shape)
print(train_labels.shape)

import matplotlib.pyplot as plt

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_images[i].astype(np.int8).squeeze() , cmap='gray', interpolation='none')
    plt.title("Number: %d" % train_labels[i])

# 指定优化器，损失函数和评价指标
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.0, nesterov=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
net.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 训练模型
net.fit(train_images, train_labels, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

# 指定测试数据
net.evaluate(test_images, test_labels, verbose=1)