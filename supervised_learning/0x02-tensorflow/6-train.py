#!/usr/bin/env python3
"""Train the network """


import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path='/tmp/model.ckpt'):
    """Train our network
    Args:
        X_train (numpy.ndarray):containing the training input data
        Y_train (numpy.ndarray): containing the training labels
        X_valid (numpy.ndarray): containing the validation input data
        Y_valid (numpy.ndarray): containing the validation labels
        layer_sizes (list): containing the number of nodes in each
                            layer of the network
        activations (list): containing the activation functions for each
                            layer of the network
        alpha: is the learning rate
        iterations (int): number of iterations to train over
        save_path: designates where to save the model
    Returns:
        the path where the model was saved
    """
    pass
