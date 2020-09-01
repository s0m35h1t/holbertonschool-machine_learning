#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras as Keras


def optimize_model(network, alpha, beta1, beta2):
    """ADAM optimization using keras
    Args:
        network: the model to optimize
        alpha: learning rate
        beta1: Adam optimization parameter
        beta2: second Adam optimization parameter
    Returns:
        None
    """
    ADAM = Keras.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=ADAM, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
