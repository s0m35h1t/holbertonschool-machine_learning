#!/usr/bin/env python3
"""Self Attention"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """self-attention Class"""

    def __init__(self, units):
        """Constructor
        Args:
            units is an integer representing
                the number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Call for the layer
        Args:
           s_prev is a tensor of shape (batch, units)
            containing the previous decoder hidden state
            hidden_states is a tensor of shape
                (batch, input_seq_len, units)containing
                the outputs of the encoder
        Returns: context, weights

            context is a tensor of shape (batch, units)
                that contains the context vector for the decoder
            weights is a tensor of shape
                (batch, input_seq_len, 1) that contains the attention weights


        """
        W = self.W(s_prev)[:, None, :]
        W += self.U(hidden_states)
        return tf.math.reduce_sum(W, axis=1), self.V(tf.math.tanh(W))
