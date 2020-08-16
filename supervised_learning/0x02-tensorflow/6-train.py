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
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train = create_train_op(loss, alpha)
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    s.run(tf.local_variables_initializer())
    for i in range(0, iterations):
        if not i % 100:
            print('After {} iterations:'.format(iters))
            trainloss, trainaccuracy = s.run((loss, accuracy),
                                             feed_dict={x: X_train,
                                                        y: Y_train})
            print('\tTraining Cost:', trainloss)
            print('\tTraining Accuracy:', trainaccuracy)
            validloss, validaccuracy = s.run((loss, accuracy),
                                             feed_dict={x, X_valid,
                                                        y: Y_valid})
            print('\tValidation Cost:', validloss)
            print('\tValidation Accuracy:', validaccuracy)
        s.run(train, feed_dict={x: X_train, y: Y_train})
    if not iterations % 100:
        print('After {} iterations:'.format(iterations))
        trainloss, trainaccuracy = s.run((loss, accuracy),
                                         feed_dict={x: X_train,
                                                    y: Y_train})
        print('\tTraining Cost:', trainloss)
        print('\tTraining Accuracy:', trainaccuracy)
        validloss, validaccuracy = s.run((loss, accuracy),
                                         feed_dict={x: X_valid,
                                                    y: Y_valid})
        print('\tValidation Cost:', validloss)
        print('\tValidation Accuracy:', validaccuracy)
    saver = tf.train.Saver()
    return saver.save(s, save_path)
