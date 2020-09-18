#!/usr/bin/env python3
"""0x08-deep_cnns"""

import tensorflow.keras as Keras


def projection_block(A_prev, filters, s=2):
    """
    Args:
        A_prev: the output from the previous layer
        filters: tuple or list containing the following filters
                 F11: the number of filters in the 1st 1x1 convolution
                 F3: the number of filters in the 3x3 convolution
                 F12: the number of filters in the 2nd 1x1 convolution
        s: stride of the first convolution in both the main path and
           the shortcut connection
    Returns:
        the activated output of the projection block
    """
    init = Keras.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters

    conv1 = Keras.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                                padding='same',
                                kernel_initializer=init)(A_prev)

    batch_c1 = Keras.layers.BatchNormalization(axis=3)(conv1)

    relu1 = Keras.layers.Activation('relu')(batch_c1)
    conv2 = Keras.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                                kernel_initializer=init)(relu1)

    batch_c2 = Keras.layers.BatchNormalization(axis=3)(conv2)

    relu2 = Keras.layers.Activation('relu')(batch_c2)
    conv3 = Keras.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                                kernel_initializer=init)(relu2)

    conv1_proj = Keras.layers.Conv2D(filters=F12, kernel_size=1, strides=s,
                                     padding='same',
                                     kernel_initializer=init)(A_prev)

    batch3 = Keras.layers.BatchNormalization(axis=3)(conv3)

    batch4 = Keras.layers.BatchNormalization(axis=3)(conv1_proj)
    return Keras.layers.Activation('relu')(Keras.layers.Add()
                                           ([batch3, batch4]))
