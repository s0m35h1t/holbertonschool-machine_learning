#!/usr/bin/env python3
"""Scaled Dot Product Attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention
    Args:
        Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix
        K is a tensor with its last two dimensions as
            (..., seq_len_v, dk) containing the key matrix
        V is a tensor with its last two dimensions as
            (..., seq_len_v, dv) containing the value matrix
        mask is a tensor that can be broadcast into
            (..., seq_len_q, seq_len_v) containing the optional mask,
            or defaulted to None
    Returns: output, weights

        outputa tensor with its last two dimensions as (..., seq_len_q, dv)
            containing the scaled dot product attention
        weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v) containing the attention weights
    """
    m = tf.matmul(Q, K, transpose_b=True)

    cast = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention = m / tf.math.sqrt(cast)

    if mask is not None:
        scaled_attention += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention, axis=-1)

    return tf.matmul(weights, V), weights
