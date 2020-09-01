#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras as Keras


def save_config(network, filename):
    """Save the configuration model in JSON format
    Args:
        network: model whose configuration should be saved
        filename: path of the file that the configuration should be saved to
    Returns:
        None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """Load the configuration model in JSON format
    Args:
        filename: path of the file containing the model’s configuration
                  in JSON format
    Returns:
        The loaded model
    """
    with open(filename, 'r') as f:
        network_config = f.read()
    return Keras.models.model_from_json(network_config)
