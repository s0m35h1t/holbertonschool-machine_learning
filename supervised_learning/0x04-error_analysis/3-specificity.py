#!/usr/bin/env python3
"""Calculates the specificity for
each class in a confusion matrixx
"""

import numpy as np


def specificity(confusion):
    """Calculates the specificity for
    each class in a confusion matrix
    Args:
        confusion (numpy.ndarray): of shape
                    (classes, classes)
    Returns:
            (numpy.ndarray): of shape (classes,)
            containing the specificity of each class
    """
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - (fp + fn + tp)

    return tn / (tn + fp)
