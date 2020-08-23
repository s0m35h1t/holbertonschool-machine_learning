#!/usr/bin/env python3
"""Normalize input data"""


import numpy as np


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix
    Args:
        X (numpy.ndarray): of shape (d, nx) to normalize
    Returns:
        The normalized X matrix
    """
    return (X - m) / s