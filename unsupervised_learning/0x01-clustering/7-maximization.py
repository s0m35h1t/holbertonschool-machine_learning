#!/usr/bin/env python3
"""contains the maximization function"""

import numpy as np


def maximization(X, g):
    """
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    probs = np.sum(g, axis=0)
    tester = np.ones((n,))
    if not np.isclose(probs, tester).all():
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        pi[i] = np.sum(g[i]) / n

        m[i] = np.matmul(g[i], X) / np.sum(g[i])

        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
