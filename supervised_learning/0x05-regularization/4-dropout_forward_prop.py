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
        L: number of ls in the network
        keep_prob: probability that a node will be kept
        All ls except the last should use the tanh activation function
    Returns:
        cache
    """
    cache = {}
    cache['A0'] = X
    for l in range(L):
        W = weights["W" + str(l + 1)]
        A = cache["A" + str(l)]
        B = weights["b" + str(l + 1)]
        Z = np.matmul(W, A) + B
        dropout = np.random.rand(Z.shape[0], Z.shape[1])
        dropout = np.where(dropout < keep_prob, 1, 0)
        if l == L - 1:
            softmax = np.exp(Z)
            cache["A" + str(l + 1)] = (softmax / np.sum(softmax, axis=0,
                                                        keepdims=True))
        else:
            tanh = np.tanh(Z)
            cache["A" + str(l + 1)] = tanh
            cache["D" + str(l + 1)] = dropout
            cache["A" + str(l + 1)] *= dropout
            cache["A" + str(l + 1)] /= keep_prob
    return cache
