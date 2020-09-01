#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import Keras


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train a model using keras
    Args:
        network: model to train
        data ((numpy.ndarray):): of shape (m, nx) containing the input data
        labels: one-hot (numpy.ndarray): of shape (m, classes) containing
                the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs (int):er of passes through data for mini-batch gradient descent
        verbose (bool): that determines if output should be printed during
                 training
        shuffle (bool): that determines whether to shuffle the batches every
                 epoch.
    Returns:
        History object generated, after training the model
    """
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle)
