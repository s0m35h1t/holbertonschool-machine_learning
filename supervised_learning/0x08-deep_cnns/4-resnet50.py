#!/usr/bin/env python3
"""0x08-deep_cnns"""

import tensorflow.keras as Keras

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ResNet-50 architecture
    Args:
        None`
    Returns:
        the keras model
    """
    X = Keras.Input(shape=(224, 224, 3))
    init = Keras.initializers.he_normal()

    conv1 = Keras.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                                strides=2, kernel_initializer=init)(X)

    batch1 = Keras.layers.BatchNormalization()(conv1)

    relu1 = Keras.layers.Activation('relu')(batch1)

    pool_1 = Keras.layers.MaxPool2D(
        pool_size=3, strides=2, padding='same')(relu1)

    prj_conv1 = projection_block(pool_1, [64, 64, 256], 1)

    id_conv2_2 = identity_block(prj_conv1, [64, 64, 256])
    id_conv2_3 = identity_block(id_conv2_2, [64, 64, 256])

    prj_conv2 = projection_block(id_conv2_3, [128, 128, 512])

    id_conv3_1 = identity_block(prj_conv2, [128, 128, 512])
    id_conv3_2 = identity_block(id_conv3_1, [128, 128, 512])
    id_conv3_3 = identity_block(id_conv3_2, [128, 128, 512])

    prj_conv3 = projection_block(id_conv3_3, [256, 256, 1024])

    id_conv4_1 = identity_block(prj_conv3, [256, 256, 1024])
    id_conv4_2 = identity_block(id_conv4_1, [256, 256, 1024])
    id_conv4_3 = identity_block(id_conv4_2, [256, 256, 1024])
    id_conv4_4 = identity_block(id_conv4_3, [256, 256, 1024])
    id_conv4_5 = identity_block(id_conv4_4, [256, 256, 1024])

    prj_conv4 = projection_block(id_conv4_5, [512, 512, 2048])

    id_conv5_1 = identity_block(prj_conv4, [512, 512, 2048])
    id_conv5_2 = identity_block(id_conv5_1, [512, 512, 2048])

    avg_pool = Keras.layers.AveragePooling2D(pool_size=7,
                                             padding='same')(id_conv5_2)

    FC = Keras.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(avg_pool)

    model = Keras.models.Model(inputs=X, outputs=FC)

    return model
