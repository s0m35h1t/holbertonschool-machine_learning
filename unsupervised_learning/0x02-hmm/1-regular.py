#!/usr/bin/env python3
"""Regular Chains"""

import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain
    Args:
    Returns:
        P is a is a square 2D numpy.ndarray of shape (n, n) representing
        the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain

    Returns:
        a numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    if P.shape[0] != P.shape[1] or np.sum(P, axis=1).all() != 1:
        return None

    w, v = np.linalg.eig(P.T)
    idx = np.where(np.isclose(w, 1))

    idx = idx[0][0] if len(idx[0]) else None
    if idx is None:
        return None

    steady = v[:, idx]

    if any(np.isclose(steady, 0)):
        return None

    steady = steady / np.sum(steady)

    return steady[np.newaxis, :]
