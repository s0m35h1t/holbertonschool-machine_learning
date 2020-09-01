#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras as Keras


def save_model(network, filename):
    """Save a model
    Args:
        network: model to save
        filename: the path of the file that the model should be saved to
    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """Load a model
    Args:
        filename: The path of the file that the model should be loaded from
    Returns:
        the loadel model
    """
    return keras.models.load_model(filename)
