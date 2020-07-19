#!/usr/bin/env python3
"""Cocatinate two matrices """


def m_shape(matrix):
    """ calculates the matrix shape
    Args:
        Matrix (np.array)
    Returns:
        M (list): Matrix shape
    """
    if not matrix:
        return None
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + m_shape(matrix[0])


def rec_cat(mat1, mat2, axis, i):
    """Concat two matrices recrusive way
    Args;
        mat1 : Matrix 1
        mat2 : Matrix 2
        axis : aixs to concat belong
        i : depth
    """
    M = []

    if i == axis:
        return mat1 + mat2

    for i in range(len(mat1)):
        M.append(rec_cat(mat1[i], mat2[i], axis, i + 1))
    return M


def cat_matrices(mat1, mat2, axis=0):
    """Adds two n-dimensional matrices
    Args:
        mat1 (numpy.ndarray): M1 matrix.
        mat2 (numpy.ndarray): M2 matrix.
    Returns:
        numpy.ndarray: concatinated matrix
    """
    m1_shape = m_shape(mat1).pop(axis)
    m2_shape = m_shape(mat2).pop(axis)

    if m1_shape != m2_shape or axis >= len(m1_shape) + 1:
        return None
    return rec_cat(mat1, mat2, axis, 0)
