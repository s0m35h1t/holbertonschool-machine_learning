#!/usr/bin/env python3
'''Add Matrix'''


def add_matrices2D(mat1, mat2):
    """ Add 2D-matrix element-wise
    Args:
        mat1 (list): M1 NxM matrix
        mat2 (list): M2 NxM matrix
    Returns:
        list: NxM matrix
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
