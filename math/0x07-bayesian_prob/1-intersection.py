#!/usr/bin/env python3

"""Likelihood"""

import numpy as np


def likelihood(x, n, P):
    """Patient who takes this drug will develop
    severe side effects.
    Args:
        x follows a binomial distribution.
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
        probabilities
        of developing severe side effects`
    Returns:
        (numpy.ndarray) with the likelihood of obtaining the data,
        x and n, for each probability in P, respectively
    """

    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    com = np.math.factorial(n)/(np.math.factorial(x) * np.math.factorial(n-x))

    return com * pow(P, x) * pow(1 - P, n - x)


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining this data with
    the various hypothetical probabilities
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
        (numpy.ndarray) containing the intersection of
        obtaining x and n with each probability in P, respectively
    """

    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr
