#!/usr/bin/env python3
"""Script to update wieghts and biases of a DNN
    with gradient descent and L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function to implement L2 regulatization using gradient descent
    Args:
        Y: one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data
            classes: is the number of classes
            m: is the number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    Returns: Cost of the network accounting for L2 regularization
    """
    m = len(Y[0])
    cache["dz{}".format(L)] = cache["A{}".format(L)] - Y
    cache["dw{}".format(L)] = cache["dz{}".format(L)] @ cache["A{}".format(L - 1)].T / m + lambtha
    cache["db{}".format(L)] = np.sum(cache["dz{}".format(L)], axis=1, keepdims=True) / m
    L = L + 1
    for i in range(1, L - 1):
        cache["dz{}".format(L - i - 1)] = weights["W{}".format(L - i)].T @ cache["dz{}".format(L - i)] * cache["A{}".format(L - i - 1)]
        cache["dw{}".format(L - i - 1)] = cache["dz{}".format(L - i - 1)] @ (cache["A{}".format(L - i - 2)]).T / m + lambtha / m * weights["W{}".format(L-1-i)]
        cache["db{}".format(L - i - 1)] = np.sum(cache["dz{}".format(L - i - 1)], axis=1, keepdims=True) / m
        weights["b{}".format(L - i)] = weights["b{}".format(L - i)] - (alpha * cache["db{}".format(L - i)])
        weights["W{}".format(L - i)] = weights["W{}".format(L - i)] - (alpha * cache["dw{}".format(L - i)])