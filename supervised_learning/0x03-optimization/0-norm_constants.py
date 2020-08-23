#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix
"""
import numpy as np


def normalization_constants(X):
    """calculates the normalization
    Args:
        X (numpy.ndarray): shape (m, nx) to normalize 
    Returns:
        the mean and variance of each feature
    """
    return X.mean(axis=0), X.std(axis=0)
