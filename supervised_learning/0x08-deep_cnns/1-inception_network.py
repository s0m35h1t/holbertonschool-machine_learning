#!/usr/bin/env python3
"""0x08-deep_cnns"""

import tensorflow.keras as Keras

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Inception network
    Agrgs:
        None
    Returns:
        modethe keras modell
    """
    activation = 'relu'
    X = Keras.Input(shape=(224, 224, 3))
    init = Keras.initializers.he_normal()

    conv_1 = Keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2, 2),
                                 padding='same', activation=activation,
                                 kernel_initializer=init)(X)
    max_pool_1 = Keras.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                           padding='same')(conv_1)
    conv_2P = Keras.layers.Conv2D(filters=64, kernel_size=1, padding='valid',
                                  activation=activation,
                                  kernel_initializer=init)(max_pool_1)
    conv_2 = Keras.layers.Conv2D(filters=192, kernel_size=3, padding='same',
                                 activation=activation,
                                 kernel_initializer=init)(conv_2P)
    max_pool_2 = Keras.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                           padding='same')(conv_2)

    i_block_1 = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    i_block_2 = inception_block(i_block_1, [128, 128, 192, 32, 96, 64])

    max_pool_3 = Keras.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                           padding='same')(i_block_2)

    i_block_3 = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    i_block_4 = inception_block(i_block_3, [160, 112, 224, 24, 64, 64])
    i_block_5 = inception_block(i_block_4, [128, 128, 256, 24, 64, 64])
    i_block_6 = inception_block(i_block_5, [112, 144, 288, 32, 64, 64])
    i_block_7 = inception_block(i_block_6, [256, 160, 320, 32, 128, 128])

    max_pool_4 = Keras.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                           padding='same')(i_block_7)

    i_block_8 = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    i_block_9 = inception_block(i_block_8, [384, 192, 384, 48, 128, 128])

    avg_pool = Keras.layers.AveragePooling2D(pool_size=[7, 7], strides=(1, 1),
                                             padding='valid')(i_block_9)

    dropout = Keras.layers.Dropout(.4)(avg_pool)

    FC = Keras.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(dropout)

    return Keras.models.Model(inputs=X, outputs=FC)
