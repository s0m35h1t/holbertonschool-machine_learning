#!/usr/bin/env python3
"""Define: Deep Neural Network Class """


import numpy as np


class DeepNeuralNetwork:
    """ deep neural network class """

    def __init__(self, nx, layers):
        """ DEep Neural Network init
        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        for layer in layers:
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {"W1": np.random.randn(layers[0], nx) *
                        np.sqrt(2 / nx),
                        "b1": np.zeros((layers[0], 1))}
        for layer, size in enumerate(layers[1:], 2):
            cur = "W" + str(layer)
            self.weights[cur] = (np.random.randn(size, layers[layer - 2]) *
                                 np.sqrt(2 / layers[layer - 2]))
            cur = "b" + str(layer)
            self.weights[cur] = np.zeros((layers[layer - 1], 1))
