#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import Keras


def save_weights(network, filename, save_format="h5"):
    """
    Function to save the weights model
    Args:
        network: model to save
        filename: the path of the file that the weights should be saved to
        save_format: format in which the weights should be saved
    Returns:None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Function to load the weights model
    Args:
        network: model whose weights should be saved
        filename: The path of the file that the weights should be loaded from
    Returns:None
    """
    network.load_weights(filename)
    return None
