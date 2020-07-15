#!/usr/bin/env python3
"""Matrix concatenation"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices. Depends on the types axis.
    Args:
        mat1 (list): M1 NxM matrix.
        mat2 (list): M2 NxM matrix.
        axis (int): 0 concatenates rows, 1 concatenates cols.
    Returns:
        list: concatenated matrix.
    """
    if not axis and len(mat1[0]) != len(mat2[0]):
        return None
    if axis and len(mat1) != len(mat2):
        return None
    return [i.copy() for i in mat1] +\
        [i.copy() for i in mat2] if axis == 0 \
        else [mat1[i].copy() + mat2[i].copy()
              for i in range(len(mat1))]
