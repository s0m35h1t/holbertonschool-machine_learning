#!/usr/bin/env python3
"""Converts one-hot matrix into a vector of labels """


import numpy as np


def one_hot_decode(one_hot):
    """ Decode one-hot coded numpy.ndarray with shape
    Args:
        one_hot (numpy.ndarray): M with shape (classes, m)
    Retruns:
        numpy.ndarray with shape (m, ) containing
        the numeric labels for each example, or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception as e:
        return None
