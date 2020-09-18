#!/usr/bin/env python3
"""0x08-deep_cnns"""

import tensorflow.keras as Keras


def dense_block(X, nb_filters, growth_rate, layers):
    """Create a dense block
    Args:
        X: the output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block
    Returns:
        The concatenated output of each layer within the Dense
        Block and the number of filters within the concatenated
        outputs, respectively
    """
    init = Keras.initializers.he_normal()

    for i in range(layers):

        batch1 = Keras.layers.BatchNormalization()(X)

        relu1 = Keras.layers.Activation('relu')(batch1)

        bottleneck = Keras.layers.Conv2D(filters=4*growth_rate,
                                         kernel_size=1, padding='same',
                                         kernel_initializer=init)(relu1)

        batch2 = Keras.layers.BatchNormalization()(bottleneck)
        relu2 = Keras.layers.Activation('relu')(batch2)

        X_conv = Keras.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                     padding='same',
                                     kernel_initializer=init)(relu2)
        nb_filters += growth_rate
        x = Keras.layers.concatenate([X, X_conv])
    return x, nb_filters
