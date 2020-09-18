#!/usr/bin/env python3
"""0x08-deep_cnns"""

import tensorflow.keras as Keras


def transition_layer(X, nb_filters, compression):
    """Create  a transition layer
    Args:
        X: the output from the previous layer
        nb_filters: integer representing the number
                    of filters in X
        compression: compression factor for the transition layer
    Returns:
        The output of the transition layer and the number
        of filters within the output, respectively
    """
    init = Keras.initializers.he_normal()
    nfilter = int(nb_filters * compression)

    batch1 = Keras.layers.BatchNormalization()(X)

    relu1 = Keras.layers.Activation('relu')(batch1)

    conv = Keras.layers.Conv2D(filters=nfilter,
                               kernel_size=1, padding='same',
                               kernel_initializer=init)(relu1)

    return Keras.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='same')(conv), nfilter
