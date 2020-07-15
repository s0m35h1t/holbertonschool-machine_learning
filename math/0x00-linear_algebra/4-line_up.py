#!/usr/bin/env python3
'''Add Arrays'''


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise
    Args:
        arr1 (list): first array
        arr2 (list): second array
    Returns:
        list: n-size array
    """
    if len(arr1) != len(arr2):
        return None
    return [[arr1[i] + arr2[i]] for i in range(len(arr1))]
