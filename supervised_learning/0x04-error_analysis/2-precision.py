#!/usr/bin/env python3
"""calculates the precision for each class in a confusion matrixx
"""

import numpy as np


def precision(confusion):
    """calculates the precision for each
    class in a confusion matrix
    Args:
        confusion: numpy.ndarray of shape
                    (classes, classes)
    Returns:
        (numpy.ndarray): of shape (classes,)
        containing the precision of each class
    """
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    return tp / (tp + fp)
