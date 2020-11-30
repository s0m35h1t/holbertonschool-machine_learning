#!/usr/bin/env python3
"""Convolutional Autoencodern"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder
    Args:
        input_dims is a tuple of integers containing the dimensions
            of the model input
        filters is a list containing the number of filters
            for each convolutional layer in the encoder, respectively
            the filters should be reversed for the decoder

        latent_dims is a tuple of integers containing the dimensions
            of the latent space representation
        Each convolution in the encoder should use a kernel size
            of (3, 3) with same padding and relu activation,
            followed by max pooling of size (2, 2)
        Each convolution in the decoder, except for the last two,
            should use a filter size of (3, 3) with same padding
            and relu activation, followed by upsampling of size (2, 2)
            The second to last convolution should instead use valid padding
            The last convolution should have the same number of filters
            as the number of channels in input_dims with sigmoid activation
            and no upsampling

    Returns: encoder, decoder, auto

            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model

    """
    e_inputs = keras.Input((input_dims,))
    d_inputs = keras.Input((latent_dims,)))

    encoder = e_inputs
    for f in filters:
        encoder = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same')(encoder)
        encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = d_inputs
    for i in reversed(range(1, len(filters))):
        decoder = keras.layers.Conv2D(
            filters[i], (3, 3), activation='relu', padding='same')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(filters[0],
                                  (3, 3), activation='relu',
                                  padding='valid')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                  activation='sigmoid',
                                  padding='same')(decoder)

    encoder = keras.Model(e_inputs, encoder)
    decoder = keras.Model(d_inputs, decoder)

    auto = keras.Model(e_inputs, decoder(encoder(e_inputs)))
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
