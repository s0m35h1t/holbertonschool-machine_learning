#!/usr/bin/env python3
"""Adds two matrices """


def add_matrices(mat1, mat2):
    """Adds two n-dimensional matrices
    Args:
        mat1 (numpy.ndarray): M1 matrix.
        mat2 (numpy.ndarray): M2 matrix.
    Returns:
        numpy.ndarray: matrix or None
    """
    try:
        if len(mat1) != len(mat2):
            raise ValueError
        if isinstance(mat1[0], list) and isinstance(mat2[0], list):
            return list(map(add_matrices, mat1, mat2)) \
                if list(map(add_matrices, mat1, mat2))[0] else None
        return [i + j for i, j in zip(mat1, mat2)]
    except ValueError:
        return None
