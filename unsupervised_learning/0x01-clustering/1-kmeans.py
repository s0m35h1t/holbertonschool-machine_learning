#!/usr/bin/env python3
"""Initialize K-meansn"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Initializes cluster centroids for K-means
    Args:
        X is a numpy.ndarray of shape (n, d) containing
        the dataset that will be used for K-means clustering

            n is the number of data points
            d is the number of dimensions for each data point

        k is a positive integer containing the number of clusters
    Returns:
        a numpy.ndarray of shape (k, d) containing
        the initialized centroids for each cluster,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    _, d = X.shape
    amin, amax = np.amin(X, axis=0), np.amax(X, axis=0)

    C = np.random.uniform(amin, amax, (k, d))
    C_old = np.copy(C)

    X_ = X[:, :, np.newaxis]
    C_ = C.T[np.newaxis, :, :]
    cls = np.argmin(np.linalg.norm(X_ - C_, axis=1), axis=1)

    for i in range(iterations):
        for j in range(k):
            idx = np.where(cls == j)
            if len(idx[0]) == 0:
                C[j] = np.random.uniform(amin, amax, (1, d))
            else:
                C[j] = np.mean(X[idx], axis=0)

        X_ = X[:, :, np.newaxis]
        C_ = C.T[np.newaxis, :, :]
        cls = np.argmin(np.linalg.norm(X_ - C_, axis=1), axis=1)

        if (C == C_old).all():
            return C, cls
        C_old = np.copy(C)

    return C, cls
