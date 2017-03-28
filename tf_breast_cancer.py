import tensorflow as tf
import numpy as np
from numpy import genfromtxt

dataset = genfromtxt('/home/kaas/Projects/exercise-breast-cancer/res/breast-cancer-wisconsin.data', delimiter=',')

# Remove rows containing nan values
dataset = dataset[~np.isnan(dataset).any(axis=1)]

# Update output values to 0 and 1 (because the sigmoid function outputs between 0 and 1)
dataset[:, -1] = (dataset[:, -1] / 2) - 1

input_data = np.delete(dataset, [0, 10], axis=1)
output_data = np.asarray([[output] for output in dataset[:, -1]])


x = tf.placeholder(tf.float32, [None, 9])
W_1 = tf.Variable(tf.random_normal([9, 9]))
b_1 = tf.Variable(tf.random_normal([9, ]))
W_2 = tf.Variable(tf.random_normal([9, 1]))
b_2 = tf.Variable(tf.random_normal([1, ]))

input_layer = tf.placeholder(tf.float32, [None, 9])
hidden_layer = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
output_layer = tf.nn.softmax(tf.matmul(hidden_layer, W_2) + b_2)

y = output_layer
y_ = tf.placeholder(tf.float32, [None, 1])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
ss_cost = tf.reduce_sum(tf.square(tf.subtract(y, y_)))

train_step = tf.train.AdamOptimizer(0.05).minimize(ss_cost)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
for _ in range(1000):

    for (x_data, y_data) in zip(input_data, output_data):
        x_data_veel_beter = [x_data.tolist()]

        y_data_veel_beter = [y_data.tolist()]
        sess.run(train_step, feed_dict={x: x_data_veel_beter, y_:y_data_veel_beter})


    print(sess.run(ss_cost, feed_dict={x: input_data, y_: output_data}))