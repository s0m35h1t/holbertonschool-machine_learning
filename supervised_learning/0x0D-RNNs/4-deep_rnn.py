#!/usr/bin/env python3
"""Deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray of shape
          (l, m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    T, m, _ = X.shape
    layers, _, h = h_0.shape
    H = np.zeros((T + 1, layers, m, h))
    H[0] = h_0
    Y = np.zeros((T, m, rnn_cells[-1].Wy.shape[1]))

    for t in range(T):
        H[t + 1, 0], Y[t] = rnn_cells[0].forward(H[t, 0], X[t])
        for l in range(1, layers):
            H[t + 1, l], Y[t] = rnn_cells[l].forward(H[t, l],
                                                     H[t + 1, l - 1])
    return H, Y
