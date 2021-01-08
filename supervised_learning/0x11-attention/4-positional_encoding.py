#!/usr/bin/env python3
"""Calculate positional encoding for a transformer"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer
    Args:
        max_seq_len is an integer representing
            the maximum sequence length
            dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectorsr"""

    angle_rates = 1 / \
        np.power(
            10000, (2 * (np.arange(dm)[np.newaxis, :]//2)) / np.float32(dm))
    angle_rads = np.arange(max_seq_len)[:, np.newaxis] * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
