#!/usr/bin/env python3
""" Cumulative N-gram BLEU score """
import numpy as np


def c_p(sentence, p):
    """Calculates the nbr that a pattern is in a sentence.
        Args:
            sentence: sentence to search.
            p: pattern to be found.
        Return: number of times the pattern
            is in the sentence.
    """
    i = 0
    for idx in range(len(sentence) - len(p) + 1):
        if sentence[idx:idx+len(p)] == p:
            i += 1
    return i


def cumulative_bleu(references, sentence, n):
    """ calculates the cumulative n-gram BLEU score for a sentence.
    Args:
        references is a list of reference translations
            each reference translation is a list
            of the words in the translation
        sentence is a list containing the model
            proposed sentence
        n is the size of the largest n-gram to use for evaluation
        All n-gram scores should be weighted evenly
    Returns: the cumulative n-gram BLEU score
    """
    bleus = []
    for j in range(1, n + 1):
        count = clips = 0
        for i in range(len(sentence) - j + 1):
            ngram = sentence[i:i+j]
            count += c_p(sentence, ngram)
            maxi = 0
            for ref in references:
                cp = c_p(ref, ngram)
                if cp > maxi:
                    maxi = cp
            clips += maxi
        bleus.append(clips / count)

    mt = len(sentence)
    ref = len(references[np.argmin(
        [abs(len(r) - mt) for r in references])])
    if mt > ref:
        b = 1
    else:
        b = np.exp(1 - ref / mt)

    return b * np.exp(np.sum(np.log(bleus) / n)))
