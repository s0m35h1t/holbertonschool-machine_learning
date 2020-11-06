#!/usr/bin/env python3
"""contains the gmm function"""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
    Retruns:
        pi, m, S, clss, bic
            pi is a numpy.ndarray of shape (k,) containing
            the cluster priors
            m is a numpy.ndarray of shape (k, d) containing
            the centroid means
            S is a numpy.ndarray of shape (k, d, d) containing
            the covariance matrices
            clss is a numpy.ndarray of shape (n,) containing
            the cluster indices for each data point
            bic is a numpy.ndarray of shape (kmax - kmin + 1) containing
            the BIC value for each cluster size tested

    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)

    fi = GMM.fit(X)
    return fi.weights_, fi.means_, fi.covariances_, GMM.predict(X), GMM.bic(X)
