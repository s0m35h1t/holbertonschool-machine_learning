#!/usr/bin/env python3
"""Sparse Autoencoden"""

import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder
    Args:
        input_dims is an integer containing the dimensions
            of the model input
        hidden_layers is a list containing the number
            of nodes for each hidden layer in the encoder, respectively
            the hidden layers should be reversed for the decoder

        latent_dims is an integer containing the dimensions
            of the latent space representation
        lambtha is the regularization parameter used for L1 regularization
            on the encoder output
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model
    """
    e_inputs = keras.Input((input_dims,))
    d_inputs = keras.Input((latent_dims,))

    encoder = e_inputs
    for hl in hidden_layers:
        encoder = keras.layers.Dense(hl, activation='relu')(encoder)
    encoder = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(encoder)

    decoder = d_inputs
    for hl in reversed(hidden_layers):
        decoder = keras.layers.Dense(hl, activation='relu')(decoder)
    decoder = keras.layers.Dense(input_dims, activation="sigmoid")(decoder)

    encoder = keras.Model(e_inputs, encoder)
    decoder = keras.Model(d_inputs, decoder)

    auto = keras.Model(e_inputs, decoder(encoder(e_inputs)))
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
