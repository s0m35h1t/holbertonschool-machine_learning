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
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        """self.b1 = [[0] for i in [nodes] * nodes]"""
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def W2(self):
        return self.__W2

    @property
    def b1(self):
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """calculates forward propagation for neural network
        Args:
            X (numpy.ndarray): (nx, m) that contains the input data
        Retruns:
            the private attributes __A1 and __A2, respectively
        """
        self.__A1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-1 * self.__A1))
        self.__A2 = (np.dot(self.__W2, self.__A1) + self.b2)
        self.__A2 = 1 / (1 + np.exp(-1 * self.__A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates cost of the neural network
                Args:
            Y (numpy.ndarray): (1, m) that contains the correct labels
                                      for the input data
            A (numpy.ndarray): (1, m) containing the activated output of
                                      the neuron for each example
        Returns:
            the cost
        """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """Evalutes the neural network
        Args:
            X (numpy.ndarray ): (nx, m) that contains the input data
            Y (numpy.ndarray ):  (1, m) that contains the correct
                                labels for the input data
        Returns:
            the neuron s prediction and the cost of the network"""
        return (self.forward_prop(X)[1].round().astype(int),
                self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates pass of gradient descent on the neuron
        Args:
            X (numpy.ndarray): (nx, m) that contains the input data
            Y (numpy.ndarray): (1, m) that contains the correct labels
                               for the input data
            A (numpy.ndarray): (1, m) containing the activated output
                                of the neuron for each example
            Alpha (float): is the learning rate
        Returns:
            (None): Updates the private attributes __W and __b
        """
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dz2 = (A2 - Y)
        self.__W2 -= alpha * np.dot(dz2, A1.T) / A1.shape[1]
        self.__b2 -= alpha * dz2.mean(axis=1, keepdims=True)
        self.__W1 -= alpha * np.dot(dz1, X.T) / X.shape[1]
        self.__b1 -= alpha * dz1.mean(axis=1, keepdims=True)
