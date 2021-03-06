#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Create a DNN using keras
    Args:
        nx (int): number of input features to the network
        layers list(): containing the number of nodes in each layer
                of the network
        activations (list): containing the activation functions used
                     for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns:
        Keras model
    """

    model = keras.Sequential()
    L2 = keras.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(keras.layers.Dense(layers[i], input_shape=(nx,),
                                         activation=activations[i],
                                         kernel_regularizer=L2,
                                         name='dense'))
        else:
            model.add(keras.layers.Dropout(1 - keep_prob))
            model.add(keras.layers.Dense(layers[i], activation=activations[i],
                                         kernel_regularizer=L2,
                                         name='dense_' + str(i)))
    return model
