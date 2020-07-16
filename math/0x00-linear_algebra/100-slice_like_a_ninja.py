#!/usr/bin/env python3
"""Slices a matrix along a specific axes"""


def np_slice(matrix, axes={}):
    """ Slices numpy.ndarray
    Args
        matrix (numpy.ndarray): M matrix
        axes (dict): axes slice to make
    Returns
        matrix: Sliced matrix
    """

    return matrix[tuple((slice(*axes.get(d, (None, None)))
                         for d in range(len(matrix.shape))))]
