#!/usr/bin/env python3
"""Converts numeric label vector into one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """Convert numeric label vector
    Args:
        Y (numpy.ndarray): with shape (m,) containing numeric class labels
        classes (int): the maximum number of classes found in Y
    Returns:
        a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        for q, l in enumerate(Y):
            one_hot[l][q] = 1
        return one_hot
    except Exception as e:
        return None
