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
    saver = tf.train.import_meta_graph("{}.meta".format(save_path))
    saver.restore(session, save_path)

    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    y_pred = tf.get_collection("y_pred")[0]
    acc = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]

    eval_y_pred = session.run(y_pred, feed_dict={x: X, y: Y})
    eval_accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
    eval_loss = session.run(loss, feed_dict={x: X, y: Y})

    return eval_y_pred, eval_accuracy, eval_losss
