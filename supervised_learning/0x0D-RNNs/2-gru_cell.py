#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """ Represents a gated recurrent unit """

    def __init__(self, i, h, o):
        """ Class constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
            Public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by that
            represent the weights and biases of the cell
                Wz and bz are for the update gate
                Wr and br are for the reset gate
                Wh and bh are for the intermediate hidden state
                Wy and by are for the output
            The weights should be initialized using a random normal
            distribution in the order listed above
            The weights will be used on the right side for matrix
            multiplication
            The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(0, 1, (i + h, h))
        self.Wr = np.random.normal(0, 1, (i + h, h))
        self.Wh = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Calculates Sigmoid
        Args:
            X float signmoid factor
        Returns: the sigmooid of x
        """
        return 1/(1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the
              data input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the
              previous hidden state
            The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        h = np.concatenate((h_prev, x_t), axis=1)
        z_sig = self.sigmoid(np.dot(h, self.Wz) + self.bz)
        r_sig = self.sigmoid(np.dot(h, self.Wr) + self.br)
        h_t = np.tanh(np.dot(np.concatenate(
            (r_sig * h_prev, x_t), axis=1), self.Wh) + self.bh)
        h_next = (1 - z_sig) * h_prev + z_sig * h_t
        y = np.dot(h_next, self.Wy) + self.by
        return h_next, np.exp(y) / np.exp(y).sum(axis=1, keepdims=True)
