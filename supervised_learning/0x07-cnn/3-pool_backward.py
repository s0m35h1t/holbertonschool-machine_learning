#!/usr/bin/env python3
"""Convolutional Neural Networks"""

import tensorflow as tf


def lenet5(x, y):
    """Lenet5 with tensorflow
    Args:
        x: tf.placeholder of shape (m, 28, 28, 1)
           containing the input images for the network
           m: the number of images
        y: tf.placeholder of shape (m, 10) containing
           the one-hot labels for the network
    Returns:
            a tensor for the softmax activated output,
            training operation that utilizes Adam
            optimization (with default hyperparameters),
            tensor for the loss of the network,
            tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    activation = tf.nn.relu
    conv_1 = tf.layers.conv_2D(filters=6, kernel_size=5,
                               padding='same', activation=activation,
                               kernel_initializer=init)(x)
    pool_1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_1)
    conv_2 = tf.layers.conv_2D(filters=16, kernel_size=5,
                               padding='valid', activation=activation,
                               kernel_initializer=init)(pool_1)
    pool_2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_2)
    flatten = tf.layers.Flatten()(pool_2)

    fc1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(fc1)
    fc3 = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)
    y_pred = fc3

    loss = tf.losses.softmax_cross_entropy(y, fc3)
    train = tf.train.AdamOptimizer().minimize(loss)

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    y_pred = tf.nn.softmax(y_pred)
    return y_pred, train, loss, accuracy
