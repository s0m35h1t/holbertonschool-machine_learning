#!/usr/bin/env python3
"""Transformer encoder block"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Keras layer call
        Args:
            x - a tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder block
            training - a boolean to determine if the model is training
            mask - the mask to be applied for multi head attentione
        Returns: output, weights
            outputa tensor with its last two dimensions as
                (..., seq_len_q, dm) containing the scaled
                dot product attention
            weights a tensor with its last three dimensions
                as (..., h, seq_len_q, seq_len_v) containing
                the attention weights
        """
        m, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(m, training=training)
        output = self.layernorm1(x + attn_output)
        ffn = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        ffn_output = self.dropout2(ffn(output), training=training)

        return self.layernorm2(output +
                               self.dropout2(ffn_output, training=training))
