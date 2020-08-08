#!/usr/bin/env python3
"""Define: Neuron class => single neuron performing binary classification"""

import numpy as np


class Neuron:
    """ Neuron class """

    def __init__(self, nx):
        """Init Neuron
        Args:
            nx: number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.ndarray.randn((1, nx))
        self.b = 0
        self.A = 0

    @property
    def W(self):
        """ Returns weights """
        return self.__W

    @property
    def b(self):
        """ Returns bias """
        return self.__b

    @property
    def A(self):
        """ Returns activation values """
        return self.__A
