#!/usr/bin/env python3
""" create training op for network """


import tensorflow as tf


def create_train_op(loss, alpha):
    """Create training op for network
    Args:
        loss (int): is the loss of the network’s prediction
        alpha (int): is the learning rate
    Returns:
        operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
