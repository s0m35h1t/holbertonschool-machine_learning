#!/usr/bin/env python3
'''Transpose Matrix'''


def matrix_transpose(matrix):
    """return trabsposed matrix
    Args:
        matrix (list): given matrix
    Returns:
        list: transposed matrix
    """
    return [[matrix[i][j] for i in range(len(matrix))]
            for j in range(len(matrix[0]))]
