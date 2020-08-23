#!/usr/bin/env python3
"""Creates a batch normalization layer for a
neural network in tensorflowfw"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a
    neural network in tensorflowf
    Args:
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activation function that should be
        ```used on the output of the layer
        you should use the tf.layers.Dense layer as the base layer
        with kernal initializer
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    Returns: tensor of the activated output for the layer

    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])

    return activation(tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        tf.Variable(tf.constant(0.0, shape=[n]), name='beta'),
        scale,
        1e-8
    ))
