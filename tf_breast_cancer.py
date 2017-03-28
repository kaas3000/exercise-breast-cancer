"""
Train a neural network on breast cancer data
"""

import numpy as np
import time
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

    weights = dict()
    weights['w1'] = tf.get_variable("weights_w1", (9, 9),
                                    initializer=tf.random_normal_initializer())
    weights['w2'] = tf.get_variable("weights_w2", (9, 1),
                                    initializer=tf.random_normal_initializer())

    biases = dict()
    biases['b1'] = tf.get_variable("bias_b1", (9,), initializer=tf.random_normal_initializer())
    biases['b2'] = tf.get_variable("bias_b2", (1,), initializer=tf.random_normal_initializer())

    x_size = 9
    # model
    input_layer = tf.placeholder(tf.float32, [None, x_size])
    hidden_layer = tf.nn.tanh(tf.matmul(x, weights['w1']) + biases['b1'])
    output_layer = tf.nn.softmax(tf.matmul(hidden_layer, weights['w2']) + biases['b2'])

    ss_cost = tf.reduce_sum(tf.square(tf.subtract(output_layer, y_)))

    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(ss_cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        run = "run_" + str(int(time.time()))
        file_writer = tf.summary.FileWriter(('log/' + run), sess.graph)
        pleb = 0

        tf.summary.scalar('ss_cost', ss_cost)

        print(sess.run(weights['w1']))

        n_epochs = 100
        n_batches = 2
        for index in range(n_epochs):
            print("Epoch {:d}".format(index))

            weights_summary_op = tf.summary.tensor_summary("weights_w1", weights["w1"])

            arie = tf.summary.merge_all()

            input_data_num_samples = input_data.shape[0]
            for (x_data, y_data) in zip(input_data, output_data):
                _, summary, loss_val = sess.run([train_step, arie, ss_cost], feed_dict={
                    x: x_data.reshape(1, -1),
                    y_: y_data.reshape(1, -1)
                })

                file_writer.add_summary(summary, pleb)

                pleb += 1

        print(sess.run(weights['w1']))
        print(
            sess.run(output_layer, feed_dict={
                x: input_data[0].reshape(1, -1),
            }), output_data[0].reshape(1, -1))
