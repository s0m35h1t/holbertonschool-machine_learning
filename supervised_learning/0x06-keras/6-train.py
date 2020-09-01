#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras as Keras


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Train a model using keras and validate data
    Args:
        network: model to train
        data: (numpy.ndarray): of shape (m, nx) containing the input data
        labels(numpy.ndarray): of shape (m, classes) containing
                the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: data to validate the model with, if not None
        early_stopping(bool): that indicates whether early stopping
                        should be used
        patience: patience used for early stopping
        verbose(bool): that determines if output should be printed during
                 training
        shuffle(bool): that determines whether to shuffle the batches every
                 epoch.
    Returns:
        History object generated after training the model
    """
    callback_ES = []
    ES = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                       patience=patience)
    if validation_data and early_stopping:
        callback_ES.append(ES)

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=callback_ES,
                          verbose=verbose, shuffle=shuffle,)
    return history
