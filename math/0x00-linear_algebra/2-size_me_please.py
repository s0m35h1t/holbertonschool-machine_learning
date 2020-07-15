#!/usr/bin/env python3
"""Matrix Shape"""


def matrix_shape(matrix):
    """Retruns matrix shape
    Args:
        matrix (list): matrix
    Returns:
        list: shape of the matrix
    """
    res = [len(i) for i in [matrix, matrix[0]]]
    return res if len(matrix[0]) <= 2 else res + [len(matrix[0][0])]
