#!/usr/bin/env python3
"""RNN decoder"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """Constructor
        Args:
            W - a Dense layer with units units,
                to be applied to the previous decoder hidden state
            U - a Dense layer with units units,
                to be applied to the encoder hidden states
            V - a Dense layer with 1 units,
                to be applied to the tanh of the sum of the outputs of W and U
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Call function for a layer
        Args:
            x is a tensor of shape (batch, input_seq_len)
                containing the input to the encoder
                layer as word indices within the vocabulary
            initial is a tensor of shape (batch, units) containing
                the initial hidden state
        Returns: outputs, hidden
            outputs is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
            hidden is a tensor of shape (batch, units) containing
                the last hidden state of the encoder
        """
        attention = SelfAttention(s_prev.shape[1])
        ctx, weights = attention(s_prev, hidden_states)
        output, state = self.gru(
            tf.concat([tf.expand_dims(ctx, 1), self.embedding(x)], -1))
        output = tf.reshape(output, (-1, output.shape[2]))
        return (self.F(output), state)
