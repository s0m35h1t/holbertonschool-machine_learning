#!/usr/bin/env python3
"""Dropout in tensorflow"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Dropout in tensorflow
    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        keep_prob: probability that a node will be kept
    Returns:
        the new layer
    """
    drop_out = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=init,
                             kernel_regularizer=drop_out)
    return tensor(prev)
