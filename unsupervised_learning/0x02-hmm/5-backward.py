#!/usr/bin/env python3
"""The Backward Algorithmn"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation

            T is the number of observations

        Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state

            Emission[i, j] is the probability of observing j given the
            hidden state i
            N is the number of hidden states
            M is the number of all possible observations

        Transition is a 2D numpy.ndarray of shape (N, N) containing
        the transition probabilities

            Transition[i, j] is the probability of transitioning from
            the hidden state i to j

        Initial a numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state

    Returns:
        P, B, or None, None on failure

        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward
        path probabilities
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j

    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    N, _ = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    if not np.sum(Emission, axis=1).all():
        return None, None

    if not np.sum(Transition, axis=1).all() or not np.sum(Initial) == 1:
        return None, None

    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones(N)

    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[t + 1]] * B[:, t + 1], axis=1)

    return np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0]), B
