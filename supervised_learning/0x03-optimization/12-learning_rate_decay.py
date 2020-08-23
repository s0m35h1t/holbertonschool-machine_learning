#!/usr/bin/env python3
"""creates a learning rate decay operation in
tensorflow using inverse time decay:"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay operation in tensorflow
        using inverse time decay
    Args:
        alpha is the original learning rate
        decay_rate is the weight used to determine
            the rate at which alpha will decay
        global_step is the number of passes of gradient
            descent that have elapsed
        decay_step is the number of passes of gradient
            descent that should occur
            before alpha is decayed furtherr
    Returns:
        the learning rate decay operationn
    """

    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
