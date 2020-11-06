#!/usr/bin/env python3
"""PDF"""

import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function
    of a Gaussian distribution
    Args:
        X is a numpy.ndarray of shape (n, d) containing
            the data points whose PDF should be evaluated
        m is a numpy.ndarray of shape (d,) containing
            the mean of the distribution
        S is a numpy.ndarray of shape (d, d) containing
            the covariance of the distribution
    Returns:
    P, or None on failure
        P is a numpy.ndarray of shape (n,) containing
        the PDF values for each data point

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    _, d = X.shape

    inv = np.linalg.inv(S)
    S_det = np.linalg.det(S)

    diff = X.T - m[:, np.newaxis]

    density = np.exp(- np.sum(diff * np.matmul(inv, diff),
                              axis=0) / 2) / np.sqrt(((2 * np.pi) ** d) * S_det)

    return np.where(density < 1e-300, 1e-300, density)
