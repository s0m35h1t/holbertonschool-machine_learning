#!/usr/bin/env python3
"""Variational Autoencodern"""

import tensorflow.keras as keras



def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder
    Args:
        input_dims is an integer containing the dimensions of the model input
        hidden_layers is a list containing the number of nodes
        for each hidden layer in the encoder, respectively

            the hidden layers should be reversed for the decoder

        latent_dims is an integer containing the dimensions
        of the latent space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output
            the latent representation, the mean, and the log variance,
            respectively
        decoder is the decoder model
        auto is the full autoencoder model
    """
    pass
