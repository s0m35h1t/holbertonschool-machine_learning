#!/usr/bin/env python3
"""Define: Deep Neural Network Class """


import numpy as np


class DeepNeuralNetwork:
    """ Deep neural network class """

    def __init__(self, nx, layers):
        """ Deep Neural Network init
        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {"W1": np.random.randn(
            layers[0], nx), "b1": np.zeros((layers[0], 1))}
        for i in range(1, self.L):
            lwei = np.random.randn(layers[i],
                                   layers[i-1]) * np.sqrt(2/layers[i-1])
            self.weights[("W{}".format(i+1))] = lwei
            self.weights[("b{}".format(i+1))] = np.zeros((layers[i], 1))
