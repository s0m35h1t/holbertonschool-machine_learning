#!/usr/bin/env python3
"""Early stopping regularization algorithm"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Early stopping regularization algorithm
    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met
    Returns:
        (bool) of whether the network should be stopped early,
        followed by the updated count
    """
    if threshold < opt_cost - cost:
        count = 0
    else:
        count += 1
    return False, count if count != patience else True, count
