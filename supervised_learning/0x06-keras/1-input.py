#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import keras as Keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Create a DNN using keras
    Args:
        nx: number of input features to the network
        layers (list): containing the number of nodes in each
                layer of the network
        activations (list): containing the activation functions
                     used for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns:
        Keras model
    """

    inputs = keras.Input(shape=(nx,))
    L2 = keras.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = keras.layers.Dense(layers[i],
                                        activation=activations[i],
                                        kernel_regularizer=L2)(inputs)
        else:
            dropout = keras.layers.Dropout(1 - keep_prob)(output)
            output = keras.layers.Dense(layers[i], activation=activations[i],
                                        kernel_regularizer=L2)(dropout)
    return keras.models.Model(inputs=inputs, outputs=output)
