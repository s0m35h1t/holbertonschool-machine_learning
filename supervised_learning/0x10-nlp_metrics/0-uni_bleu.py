#!/usr/bin/env python3
"""Unigram BLEU scoree"""


import numpy as np


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence
    Args:
        references is a list of reference translations
            each reference translation is a list of
            the words in the translation
        sentence is a list containing the model
            proposed sentence
    Returns: the unigram BLEU score
    """
    count = clips = 0
    for i in sentence:
        count += sentence.count(i)
        maxi = 0
        for ref in references:
            if ref.count(i) > maxi:
                maxi = ref.count(i)
        clips += maxi

    mt = len(sentence)
    ref = len(references[np.argmin(
        [abs(len(r) - mt) for r in references])])
    if mt > ref:
        b = 1
    else:
        b = np.exp(1 - ref / mt)

    return b * (clips / count)
