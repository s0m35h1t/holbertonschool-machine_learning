#!/usr/bin/env python3
"""Script to normalize a batch for a DNN"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
        using batch normalizationh
    Args:
        Z is a numpy.ndarray of shape (m, n) that
            should be normalized
        gamma is a numpy.ndarray of shape (1, n)
            containing the scales used for batch normalization
        beta is a numpy.ndarray of shape (1, n)
            containing the offsets used for batch normalization
        epsilon is a small number used to avoid division
            by zerox
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Z_normalization = Z - mean / np.sqrt(variance + epsilon)

    return gamma * Z_normalization + beta
