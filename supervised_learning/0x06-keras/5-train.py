#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

import tensorflow.keras as keras


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Train a model using keras and validate data
    Args:
        network: model to train
        data ((numpy.ndarray):): of shape (m, nx) containing the input data
        labels ((numpy.ndarray):): one-hot of shape (m, classes) containing
                the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs (int): of passes through data for mini-batch gradient descent
        validation_data: data to validate the model with, if not None
        verbose (int): that determines if output should be printed during
                 training
        shuffle (int): that determines whether to shuffle the batches every
                 epoch.
    Returns:
        History object generated after training the model
    """
    validation_data = validation_data if validation_data else None

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=verbose,
                       shuffle=shuffle)
