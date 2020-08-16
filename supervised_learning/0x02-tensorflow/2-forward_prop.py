#!/usr/bin/env python3
"""Creates the forward propagation graph for the neural network:"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create basic forward propagation network
    Args:
        x: is the placeholder for the input data
        layer_sizes (list): containing the number of nodes in each
                            layer of the network
        activations (list): containing the activation functions for
                            each layer of the network
    Returns:
        the prediction of the network in tensor form
    """
    first_layer = create_layer(x, layer_sizes[0], activations[0])
    sec_layer = first_layer
    for layer in range(1, len(layer_sizes)):
        sec_layer = create_layer(sec_layer, layer_sizes[layer],
                                 activations[layer])
    return sec_layer
