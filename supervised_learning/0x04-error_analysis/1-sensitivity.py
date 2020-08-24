#!/usr/bin/env python3
"""Calculates the sensitivity for each class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix
    Args:
        confusion (numpy.ndarray): of shape
                    (classes, classes)
    Returns:
        (numpy.ndarray )of shape (classes,)
        containing the sensitivity of each class
    """
    tp = np.diag(confusion)
    fn = np.sum(confusion, axis=1) - tp
    return tp / (tp + fn)
