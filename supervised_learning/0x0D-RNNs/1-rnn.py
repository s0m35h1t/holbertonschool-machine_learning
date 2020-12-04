#!/usr/bin/env python3
"""RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN
    Args:
        X is the data to be used, given as a numpy.ndarray of
          shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray
          of shape (m, h)
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    T, m, _ = X.shape
    _, h = h_0.shape[1]
    H = np.zeros((T + 1, m, h))
    Y = np.zeros((T, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0
    for t in range(T):
        H[t + 1], Y[t] = rnn_cell.forward(H[t], X[t])

    return H, Y
