#!/usr/bin/env python3
""" Create tensor output of previous layer """


import tensorflow as tf


def create_layer(prev, n, activation):
    """Create tensorflowf layer
    Args:
        prev (): is the tensor output of the previous layer
        n (): is the number of nodes in the layer to create
        activation (): is the activation function that the layer should use
    Returns:
        the tensor output of the layer
    """
    initializer = (tf.contrib.layers.
                   variance_scaling_initializer(mode="FAN_AVG"))
    return tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=initializer)(prev)
