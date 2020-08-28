#!/usr/bin/env python3
"""implement L2 regularizationN"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """implement L2 regularization
    Args:
        cost: cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights (dict):the weights and biases
                (numpy.ndarrays) of the neural network
        L (int): number of layers in the neural network
        m (int): number of data points used
    Returns:
        cost of the network accounting for L2 regularization
    """
    norm = 0
    for k, v in weights.items():
        if k[0] == 'W':
            norm = norm + np.linalg.norm(v)
    return cost + (lambtha / (2 * m) * norm)
