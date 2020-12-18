#!/usr/bin/env python3
"""Extract tf_idf representation"""


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding:
    Args:
        sentences is a list of sentences to analyze
        vocab is a list of the vocabulary words to use for the analysis
            If None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    data = vectorizer.fit_transform(sentences)

    bow, vect = data.toarray(), vectorizer.get_feature_names()
    return TfidfTransformer().fit_transform(bow).toarray(), vect
