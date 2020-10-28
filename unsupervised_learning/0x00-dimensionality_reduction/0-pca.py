#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """Perfoms PCA on a data set
    Args:
        X: (numpy.ndarray) of shape (n, d) where
           n (int) of data points
           d (int) of dimensions in each point
           all dimensions have a mean of 0 across all data points
        var: fraction of the variance that the PCA transformation
            should maintain
    Returns: 
        weights matrix, W, that maintains var fraction of X‘s
        original variance. W is a ndarray (d, nd)
        nd is the new dimensionality o the transformed X
    """
    _, S, Vt = np.linalg.svd(X)
    sum_s = np.cumsum(S)
    sum_s = sum_s / sum_s[-1]
    r = np.min(np.where(sum_s >= var))

    V = Vt.T
    Vr = V[..., :r + 1]

    return Vr