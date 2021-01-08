#!/usr/bin/env python3
"""Multi Head Attentionr"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Calculate multi-head attention for a transformer"""

    def __init__(self, dm, h):
        """Constructor
        Args:
            dm is an integer representing the dimensionality of the model
            h is an integer representing the number of heads
        public instance attributes
            h - the number of heads
            dm - the dimensionality of the model
            depth - the depth of each attention head
            Wq - a Dense layer with dm units, used to generate the query matrix
            Wk - a Dense layer with dm units, used to generate the key matrix
            Wv - a Dense layer with dm units, used to generate the value matrix
            linear - a Dense layer with dm units, used to generate the
                    attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // self.h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batches):
        """Split the last dimension into (num_heads, depth).
        """
        rb = tf.reshape(x, (batches, -1, self.h, self.depth))
        return tf.transpose(rb, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Keras layer call"""
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        return self.linear(tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))), weights
