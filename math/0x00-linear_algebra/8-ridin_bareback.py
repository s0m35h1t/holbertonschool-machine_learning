#!/usr/bin/env python3
"""Matrix Multiplication"""


def mat_mul(mat1, mat2):
    """Performs 2D-matrix multiplication.
    Args:
        mat1 (list): M1 NxM matrix.
        mat2 (list): M2 KxP matrix.
    Returns:
        list: NxP matrix.
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a*b for a, b in zip(x, y)) for y in zip(*mat2)]
            for x in mat1]
