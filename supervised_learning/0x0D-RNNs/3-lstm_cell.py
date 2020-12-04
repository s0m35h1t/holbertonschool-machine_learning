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
            Public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
            that represent the weights and biases of the cell
                Wf and bf are for the forget gate
                Wu and bu are for the update gate
                Wc and bc are for the intermediate cell state
                Wo and bo are for the output gate
                Wy and by are for the outputst
            The weights should be initialized using a random normal
            distribution in the order listed above
            The weights will be used on the right side for matrix
            multiplication
            The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Calculates Sigmoid
        Args:
            X float signmoid factor
        Returns: the sigmooid of x
        """
        return 1/(1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data
              input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
             hidden state
            c_prev is a numpy.ndarray of shape (m, h) containing the previous
              cell state
            The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        h = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(h.dot(self.Wf) + self.bf)
        u = self.sigmoid(h.dot(self.Wu) + self.bu)
        c = np.tanh(h.dot(self.Wc) + self.bc)
        c = f * c_prev + u * c
        o_t = self.sigmoid(h.dot(self.Wo) + self.bo)
        h_t = o_t * np.tanh(c)
        y = h_t.dot(self.Wy) + self.by
        return h_t, c, np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
