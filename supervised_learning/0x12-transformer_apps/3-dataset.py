#!/usr/bin/env python3
"""Dataset"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def filter_max_length(x, y, max_length=max_len):
    """Filtring max length"""
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


class Dataset():
    """Dataset"""

    def __init__(self):
        """Constructor
        Attrs:
            data_train, which contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervided
            data_valid, which contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt is the Portuguese tokenizer created
                from the training set
            tokenizer_en is the English tokenizer created from
                the training set
        """
        egs, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                  with_info=True,
                                  as_supervised=True)
        data_train, data_valid = egs['train'], egs['validation']
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        self.data_train = data_train.map(self.tf_encode)
        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        size = metadata.splits['train'].num_examples
        self.data_train = data_train.shuffle(size).\
            padded_batch(_, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = data_valid.map(self.tf_encode)
        self.data_valid = data_valid.filter(filter_max_length).\
            padded_batch(_, padded_shapes=([None], [None]))
        self.data_valid = data_valid

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset
        Args:
            data is a tf.data.Dataset whose examples
            are formatted as a tuple (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing
                the corresponding English sentence
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tkn_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)

        tkn_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        return tkn_pt, tkn_en

    def encode(self, pt, en):
        """encodes a translation into tokens
        Args:
            pt is the tf.Tensor containing
                the Portuguese sentence
            en is the tf.Tensor containing
                the corresponding English sentence
        Returns: pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size+1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size+1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """wrapper for the encode instance method
        Args:
            Make sure to set the shape of the pt and en return tensors
        """
        rslt_pt, rsl_en = tf.py_function(self.encode, [pt, en],
                                         [tf.int64, tf.int64])

        rslt_pt.set_shape([None])
        rsl_en.set_shape([None])

        return rslt_pt, rsl_en
