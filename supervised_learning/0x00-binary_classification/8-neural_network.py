#!/usr/bin/env python3
"""Define: Neural network class """


import numpy as np


class NeuralNetwork:
    """ neural network class """

    def __init__(self, nx, nodes):
        """ nx: number of input features """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        self.__W1 = np.ndarray((nodes, nx))
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__W2 = np.ndarray((1, nodes))
        self.__W2[0] = np.random.normal(size=nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0