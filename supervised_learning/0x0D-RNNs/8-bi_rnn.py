#!/usr/bin/env python3
""" Bidirectional RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN
    Args:
        bi_cell is an instance of BidirectionalCell that will
          be used for the forward propagation
        X is the data to be used, given as a numpy.ndarray of shape
          (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state in the forward direction, given
          as a numpy.ndarray of shape (m, h)
            h is the dimensionality of the hidden state
        h_t is the initial hidden state in the backward direction, given
          as a numpy.ndarray of shape (m, h)
    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated
            hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    T, m, _ = X.shape
    _, h = h_0.shape
    Hf, Hb = np.zeros((T + 1, m, h)), np.zeros((T + 1, m, h))
    Hf[0] = h_0
    Hb[-1] = h_t
    for i, j in zip(range(T), range(T - 1, -1, -1)):
        Hf[i + 1] = bi_cell.forward(Hf[i], X[i])
        Hb[j] = bi_cell.backward(Hb[j + 1], X[j])
    H = np.concatenate((Hf[1:], Hb[0:-1]), axis=-1)
    return H, bi_cell.output(H)
