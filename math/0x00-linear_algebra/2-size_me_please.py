#!/usr/bin/env python3
"""Matrix Shape"""


def matrix_shape(matrix):
    """Retruns matrix shape
    Args:
        matrix (list): matrix
    Returns:
        list: shape of the matrix
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])
