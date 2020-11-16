#!/usr/bin/env python3
"""The Viretbi Algorithmn"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """calculates the most likely sequence of hidden states for a
    hidden markov mode
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation

            T is the number of observations

        Emission is a numpy.ndarray of shape (N, M) containing
        the emission probability
        of a specific observation given a hidden state

            Emission[i, j] is the probability of observing j given
            the hidden state i
            N is the number of hidden states
            M is the number of all possible observations

        Transition is a 2D numpy.ndarray of shape (N, N) containing
        the transition probabilities

            Transition[i, j] is the probability of transitioning from
            the hidden state i to j

        Initial a numpy.ndarray of shape (N, 1) containing
        the probability of starting in a particular hidden state

    Returns:
        path, P, or None, None on failure

            path is the a list of length T containing the most likely sequence
            of hidden states
            P is the probability of obtaining the path sequence
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
    viterbi = np.zeros((N, T))
    bp = np.zeros((N, T))

    bp[:, 0] = 0
    viterbi[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])

    for t in range(1, T):
        ab = viterbi[:, t - 1] * Transition.T
        ab_max = np.amax(ab, axis=1)
        viterbi[:, t] = ab_max * Emission[:, Observation[t]]
        bp[:, t - 1] = np.argmax(ab, axis=1)

    path = [np.argmax(viterbi[:, T - 1])] + []
    curr = np.argmax(viterbi[:, T - 1])

    for t in range(T - 2, -1, -1):
        curr = int(bp[curr, t])
        path = [curr] + path

    return path, np.amax(viterbi[:, T - 1], axis=0)
