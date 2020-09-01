#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

import tensorflow.keras as keras


def predict(network, data, verbose=False):
    """Makes a prediction using keras
    Args:
        network: network model to make the prediction with
        data: input data to make the prediction with
        verbose (bool): that determines if output should
                be printed during the prediction process
    Returns:
        The prediction of the data
    """
    return network.predict(data, verbose=verbose)
