#!/usr/bin/env python3
"""Dataset"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset
        Args:
            data is a tf.data.Dataset whose examples
            are formatted as a tuple (pt, en)
                pt is the tf.Tensor containing the Portuguese sentence
                en is the tf.Tensor containing the corresponding English sentence
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
        result_pt, result_en = tf.py_function(self.encode, [pt, en],
                                              [tf.int64, tf.int64])

        return result_pt.set_shape([None]), result_en.set_shape([None])

