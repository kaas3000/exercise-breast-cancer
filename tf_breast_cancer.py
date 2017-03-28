"""
Train a neural network on breast cancer data
"""

import numpy as np
from numpy import genfromtxt
import tensorflow as tf

if __name__ == '__main__':
    dataset = genfromtxt('res/breast-cancer-wisconsin.data', delimiter=',')

    # Remove rows containing nan values
    dataset = dataset[~np.isnan(dataset).any(axis=1)]

    # Update output values to 0 and 1 (because the sigmoid function outputs between 0 and 1)
    dataset[:, -1] = (dataset[:, -1] / 2) - 1

    input_data = np.delete(dataset, [0, 10], axis=1)
    output_data = np.asarray([[output] for output in dataset[:, -1]])

    # Setup values
    x = tf.placeholder(tf.float32, shape=[None, 9])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    with tf.variable_scope("dude-where-is-my-car"):
        weights = dict()
        weights['w1'] = tf.Variable(tf.random_normal([9, 9]))
        weights['w2'] = tf.Variable(tf.random_normal([9, 1]))

        biases = dict()
        biases['b1'] = tf.Variable(tf.random_normal([9, ]))
        biases['b2'] = tf.Variable(tf.random_normal([1, ]))

        x_size = 9
        # model
        input_layer = tf.placeholder(tf.float32, [None, x_size])
        hidden_layer = tf.nn.softmax(tf.add(
            tf.matmul(x, weights['w1']),
            biases['b1']
        ))
        output_layer = tf.nn.softmax(tf.add(
            tf.matmul(hidden_layer, weights['w2']),
            biases['b2']
        ))

        ss_cost = tf.reduce_sum(tf.square(tf.subtract(output_layer, y_)))

        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(ss_cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    tf.summary.scalar('ss_cost', ss_cost)
    arie = tf.summary.merge_all()

    file_writer = tf.summary.FileWriter('log', sess.graph)
    pleb = 0
    for index in range(100):
        print("Epoch {:d}".format(index))
        for (x_data, y_data) in zip(input_data, output_data):
            _, summary = sess.run([train_step, arie], feed_dict={
                x: x_data.reshape(1, -1),
                y_: y_data.reshape(1, -1)
            })

            file_writer.add_summary(summary, pleb)

            pleb += 1

    print(
        sess.run(output_layer, feed_dict={
            x: input_data[0].reshape(1, -1),
        }))

    file_writer.close()
