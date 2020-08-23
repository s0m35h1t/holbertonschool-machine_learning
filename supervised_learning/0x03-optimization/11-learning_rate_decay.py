#!/usr/bin/env python3

""" training operation for a neural network in tensorflow using 
the Adam optimization algorithm:r"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse
    time decay in numpyk 
    Args:
        alpha is the original learning rate
        decay_rate is the weight used to determine the rate at which alpha will decay
        global_step is the number of passes of gradient descent that have elapsed
        decay_step is the number of passes of gradient descent that should occur
            before alpha is decayed furthero
    Returns: 
        updated value for alphaa
    """
    return alpha / (1 + decay_rate * int(global_step / decay_step))
