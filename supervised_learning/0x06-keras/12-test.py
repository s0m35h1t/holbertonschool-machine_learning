#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras as Keras


def test_model(network, data, labels, verbose=True):
    """Test a model using keras
    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: are the correct one-hot labels of data
        verbose (bool): that determines if output should be printed during
                 the testing process
    Returns:
        The loss and accuracy of the model with the testing data.
    """
    return network.evaluate(data, labels, verbose=verbose)
