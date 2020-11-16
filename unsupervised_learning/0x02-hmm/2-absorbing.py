#!/usr/bin/env python3
"""Absorbing Chains"""

import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n) representing
        the standard transition matrix

            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain

    Returns:
        True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False

    if P.shape[0] != P.shape[1]:
        return False
    
    if np.sum(P, axis=1).all() != 1:
        return False

    P_diag = np.diag(P)

    if all(P_diag == 1):
        return True

    if not any(P_diag == 1):
        return False

    states = np.where(np.diag(P) == 1)
    acc = np.sum(P[states[0]], axis=0)

    for i in range(P.shape[0]):
        inter = acc * (P[i] != 0)
        if (inter == 1).any():
            acc[i] = 1
    return acc.all()
