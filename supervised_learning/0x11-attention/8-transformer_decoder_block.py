#!/usr/bin/env python3
'''Transformer Decoder'''
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    '''class DecoderBlock'''
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        '''Constructor
        Args
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layer
            drop_rate - the dropout rate
        '''
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        '''call method for kreas layer
        Args:
            x - a tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder
            encoder_output - a tensor of shape
                (batch, input_seq_len, dm)containing the output of the encoder
            training - a boolean to determine if the model is training
            look_ahead_mask - the mask to be applied to the first
                multi head attention layer
            padding_mask - the mask to be applied to the second
                multi head attention layer
        Returns:
        a tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        '''
        m, _ = self.mha1(x, x, x, look_ahead_mask)
        out = self.layernorm1(self.dropout1(m, training=training) + x)
        attn2, attn_weights_block2 = self.mha2(
            out, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        output = self.layernorm2(attn2 + out)
        ffn = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        ffn_output = ffn(output)
        ffn_output = self.dropout3(ffn_output, training=training)

        return self.layernorm3(ffn_output + output)