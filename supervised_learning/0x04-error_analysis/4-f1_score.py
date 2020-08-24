#!/usr/bin/env python3
"""calculates the F1 score of a confusion matrix"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix
    Args:
        confusion (numpy.ndarray): of shape
                    (classes, classes)
    Returns:
        ( numpy.ndarray): of shape (classes, classes)
        with row indices representing the correct labels
        and column indices representing the predicted labels
    """
    p = precision(confusion)
    s = sensitivity(confusion)
    return (p * s) * 2 / (p + s)
