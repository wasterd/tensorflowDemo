# coding=utf-8
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from numpy import random

#训练参数
learning_rate = 0.01
training_epochs = 1000
display_step = 50
logs_path = './example'

# 训练数据
train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# 定义两个变量 op 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 初始化模型里面的 Ｗ Ｂ
W = tf.Variable(random.random(), name="weight")
b = tf.Variable(random.random(), name="bias")

# 构造线性模型
pred = tf.add(tf.multiply(X,W), b) #y = WX + b

# 均方误差 RSS (计算所有误差的平均值)
cost = tf.reduce_sum(tf.pow(pred - Y , 2)) /(2 * n_samples)

"""
x = [[1,1,1]. [1,1,1]]
tf.reduce_sum(x) => 6
tf.reduce_sum(x,0) => [2,2,2]
tf.reduce_sum(x,1) => [3,3]
"""

# 梯度下降
opeimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有的变量
init = tf.global_variables_initializer()

# 创建 summmary 來观察损失值
tf.summary.scalar("loss", cost)

# 合併所有的 op
merged_summary_op = tf.summary.merge_all()

# session
with tf.Session() as sess:
    #init 初始化所有的变量
    sess.run(init)

    #将输入写入logs_path的路径下
    summary_writter = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    #开始训练
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X, train_Y):
            sess.run(opeimizer, feed_dict={X: train_X, Y: train_Y})

        #每一轮训练的时候打印一次训练结果
        if (epoch + 1) % display_step == 0:
            c, summary = sess.run([cost, merged_summary_op], feed_dict={X: train_X, Y: train_Y})
            summary_writter.add_summary(summary, epoch * n_samples)
            print("Epoch:", '%04d' %(epoch + 1), "cost:","{:.9f}".format(c),\
                  "W=",sess.run(W), "b=", sess.run(b))

    print("训练结束：！")
    train_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost = ", train_cost, "W=",sess.run(W), "b=", sess.run(b), '\n')

    #画圆
    plt.plot(train_X, train_Y, 'rx', label = 'Original Data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    plt.savefig('liner_train.png')

    # 测试数据
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("开始测试")

    testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y , 2)) /(2 * n_samples), feed_dict={X: train_X, Y: train_Y})

    print("Testing cost =", testing_cost)
    print("ABS of mean square loss difference", abs(train_cost - testing_cost))
    #画红叉
    plt.plot(test_X, test_Y, 'bo', label='Testing Data')#画点
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')#画线
    plt.legend()
    plt.show()#显示画出的结果
    plt.savefig('liner_test.png')#画的结果保存为.png格式的图片
