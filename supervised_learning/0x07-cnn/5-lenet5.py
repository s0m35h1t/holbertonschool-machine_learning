#!/usr/bin/env python3
"""Convolutional Neural Networks"""

import tensorflow.keras as K


def lenet5(X):
    """Implement Lenet-5 using keras
    Args:
        X: X is a K.Input of shape (m, 28, 28, 1)
           containing the input images for the
    Returns:
        (K.Model) compiled to use Adam optimization
        (with default hyperparameters) and accuracy
        metrics
    """
    init = K.initializers.he_normal()
    activation = 'relu'
    conv_1 = K.layers.conv_2D(filters=6, kernel_size=5,
                              padding='same', activation=activation,
                              kernel_initializer=init)(X)
    pool_1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_1)
    conv_2 = K.layers.conv_2D(filters=16, kernel_size=5,
                              padding='valid', activation=activation,
                              kernel_initializer=init)(pool_1)
    pool_2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_2)
    flatten = K.layers.Flatten()(pool_2)
    fc1 = K.layers.Dense(units=120, activation=activation,
                         kernel_initializer=init)(flatten)
    fc2 = K.layers.Dense(units=84, activation=activation,
                         kernel_initializer=init)(fc1)
    fc3 = K.layers.Dense(units=10, kernel_initializer=init,
                         activation='softmax')(fc2)
    model = K.models.Model(X, fc3)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
