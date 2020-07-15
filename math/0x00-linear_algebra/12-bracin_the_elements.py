#!/usr/bin/env python3
"""Array operations"""


def np_elementwise(mat1, mat2):
    """+. -. *, / operations between matrix.
    Args:
        mat1 (numpy.ndarray): M1 matrix.
        mat2 (numpy.ndarray): M2 matrix.
    Returns:
        numpy.ndarray: M1+M2 M1-M2, M1*M2 M1/M2
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
