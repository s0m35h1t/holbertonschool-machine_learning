#!/usr/bin/env python3
"""Script to implement dropout in a forward propagation"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that uses dropout in a forward propagation
    DNN
    Args:
        X: numpy.ndarray of shape (nx, m) containing
            the input data for the network
            nx: is the number of input features
            m: is the number of data points
        weights: dictionary of the weights and biases
        of the neural network
        L: number of layers in the network
        keep_prob: probability that a node will be kept
        All layers except the last should use the tanh activation function
    Returns:
        cache
    """
    cache = {'A0': X}
    z = {}
    for i in range(1, L+1):
        z["z{}".format(i)] = np.add(np.matmul(weights["W{}".format(
            i)], cache["A{}".format(i-1)]), weights["b{}".format(i)])
        cache["A{}".format(i)] = np.tanh(z["z{}".format(i)])
        cache["D{}".format(i)] = np.random.rand(
            cache["A{}".format(i)].shape[0], cache["A{}".format(i)].shape[1])
        cache["D{}".format(i)] = cache["D{}".format(i)] < keep_prob

        cache["A{}".format(i)] = np.multiply(
            cache["A{}".format(i)], cache["D{}".format(i)])
        cache["A{}".format(i)] = cache["A{}".format(i)]/keep_prob
    return cache
