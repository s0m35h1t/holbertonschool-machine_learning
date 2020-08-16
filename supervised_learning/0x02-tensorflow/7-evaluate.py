#!/usr/bin/env python3
"""Evaluates the output of a neural network"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network:
    Args:
        X (numpy.ndarray): containing the input data to evaluate
        Y (numpy.ndarray): containing the one-hot labels for X
        save_path (str): the location to load the model from
    Returns:
        the network’s prediction, accuracy, and loss, respectively
    """
    session = tf.Session()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(session, save_path)
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    return session.run((graph.get_tensor_by_name("layer_2/BiasAdd:0"),
                        graph.get_tensor_by_name("Mean:0"),
                        graph.get_tensor_by_name(
                            "softmax_cross_entropy_loss/value:0")),
                       feed_dict={x: X, y: Y})
