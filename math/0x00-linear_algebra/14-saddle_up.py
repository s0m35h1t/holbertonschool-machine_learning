#!/usr/bin/env python3
"""Matrix multiplication"""
import numpy as np


def np_matmul(mat1, mat2):
    """Performs n-dimensional matrix multiplication.
    Args:
        mat1 (numpy.ndarray): M1 matrix.
        mat2 (numpy.ndarray): M2 matrix.
    Returns:
        numpy.ndarray: M1 * M2.
    """
    return np.matmul(mat1, mat2)
