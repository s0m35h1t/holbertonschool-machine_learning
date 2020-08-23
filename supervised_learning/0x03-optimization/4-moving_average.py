#!/usr/bin/env python3
""" calculate weighted moving average of data set """


def moving_average(data, beta):
    """Calculate weighted moving average of data set
    Args:
        data (list): of data to calculate the moving average of
        beta (int): the weight used for the moving average
    Returns:
        (list) containing the moving averages of data
    """
    tmp = [0]
    unbiased = []
    for i, k in enumerate(data):
        tmp.append(beta * tmp[i] + (1 - beta) * k)
        unbiased.append(tmp[i + 1] / (1 - beta ** (i + 1)))
    return unbiased
