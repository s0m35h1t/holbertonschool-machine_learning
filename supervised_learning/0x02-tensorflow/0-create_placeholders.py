#!/usr/bin/env python3
""" Create Tensor placeholders for data """

import tensorflow as tf


def create_placeholders(nx, classes):
    """ Create tensor placeholders for input data and one-hot lbls
    Args:
        nx (int): the number of feature columns in our data
        classes (int): the number of classes in our classifier
    Returns:
        placeholders named x and y, respectively
            x is the placeholder for the input data to the neural network
            y is the placeholder for the one-hot labels for the input data
    """
    return (tf.placeholder(float, shape=[None, nx], name='x'),
            tf.placeholder(float, shape=[None, classes], name='y'))
