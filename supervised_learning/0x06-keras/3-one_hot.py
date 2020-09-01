#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras


def one_hot(labels, classes=None):
    """Implement one-hot using keras
    Args:
        labels: labels of the set
        classes: classes of the set
    Returns:
        The one-hot matrix
    """
    return keras.utils.to_categorical(labels, num_classes=classes)
