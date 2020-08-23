#!/usr/bin/env python3
"""Shuffle data in two matrices in the same way"""


import numpy as np


def shuffle_data(X, Y):
    """Shuffle data in two matrices
    Args:
        X (numpy.ndarray): of shape (m, nx) to shuffle
        Y (numpy.ndarray): of shape (m, ny) to shuffle 
    Returns:
        the shuffled X and Y matrices
    """
    shuffl_id = np.random.permutation(X.shape[0])
    return X[shuffl_id], Y[shuffl_id]