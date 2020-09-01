#!/usr/bin/env python3
"""builds a neural network with the Keras librarys"""

from tensorflow import Keras


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

    model = Keras.Sequential()
    L2 = Keras.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(Keras.layers.Dense(layers[i], input_shape=(nx,),
                                         activation=activations[i],
                                         kernel_regularizer=L2,
                                         name='dense'))
        else:
            model.add(Keras.layers.Dropout(1 - keep_prob))
            model.add(Keras.layers.Dense(layers[i], activation=activations[i],
                                         kernel_regularizer=L2,
                                         name='dense_' + str(i)))
    return model
