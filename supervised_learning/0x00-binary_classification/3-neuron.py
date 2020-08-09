#!/usr/bin/env python3
"""Define: Neuron class => single neuron performing binary classification"""

import numpy as np


class Neuron:
    """ Neuron class """

    def __init__(self, nx):
        """Init Neuron
        Args:
            nx (int): number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Returns weights"""
        return self.__W

    @property
    def b(self):
        """Returns bias"""
        return self.__b

    @property
    def A(self):
        """Returns activation values"""
        return self.__A

    def forward_prop(self, X):
        """Calculates forward propagation of neuron
        Args:
            X (numpy.ndarray): nx, m) that contains the input data
        Returns:
            the private attribute __A
        """
        self.__A = 1 / (1 + np.exp(-1 * (np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculates cost of the model using logistic regression
        Args:
            Y (numpy.ndarray): (1, m) that contains the correct labels
                                      for the input data
            A (numpy.ndarray): (1, m) containing the activated output of
                                      the neuron for each example
        Returns:
            the cost
        """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
