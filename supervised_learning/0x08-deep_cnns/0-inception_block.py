#!/usr/bin/env python3

import tensorflow.keras as Keras


def inception_block(A_prev, filters):
    """Create an inception block
    Args:
        A_prev: The output from the previous layer
        filters: Tuple or list containing the following filters
                 F1: is the number of filters in the 1x1 convolution
                 F3R: is the number of filters in the 1x1 convolution
                      before the 3x3 convolution
                 F3: is the number of filters in the 3x3 convolution
                 F5R: is the number of filters in the 1x1 convolution
                      before the 5x5 convolution
                 F5: is the number of filters in the 5x5 convolution
                 FPP: is the number of filters in the 1x1 convolution
                      after the max pooling
    Returns:
        the concatenated output of the inception block
    """
    activation = 'relu'
    init = Keras.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1 = Keras.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                                 activation=activation,
                                 kernel_initializer=init)(A_prev)

    conv_2P = Keras.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                                  activation=activation,
                                  kernel_initializer=init)(A_prev)

    conv_2 = Keras.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                                 activation=activation,
                                 kernel_initializer=init)(conv_2P)

    conv_3P = Keras.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                                  activation=activation,
                                  kernel_initializer=init)(A_prev)

    conv_3 = Keras.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                                 activation=activation,
                                 kernel_initializer=init)(conv_3P)

    layer_pool = Keras.layers.MaxPooling2D(pool_size=[3, 3], strides=(1, 1),
                                           padding='same')(A_prev)

    layer_pool_P = Keras.layers.Conv2D(filters=FPP, kernel_size=1,
                                       padding='same',
                                       activation=activation,
                                       kernel_initializer=init)(layer_pool)

    return Keras.layers.concatenate([conv_1, conv_2,
                                     conv_3, layer_pool_P])
