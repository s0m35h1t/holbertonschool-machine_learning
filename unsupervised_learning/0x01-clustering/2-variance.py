#!/usr/bin/env python3
"""Variancen"""

import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance
    Args
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
    Returns:
    var, or None on failure
        var is the total variance
    """
    try:
        se = np.sum(C ** 2, axis=1)[:, np.newaxis] - \
            2 * np.matmul(C, X.T) + np.sum(X ** 2, axis=1)
        return np.sum(np.amin(se, axis=0))

    except Exception:
        return None
