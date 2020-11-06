#!/usr/bin/env python3
"""contains the agglomerative function"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset
    Args:
        X is a numpy.ndarray of shape (n, d) containing
        the dataset
        dist is the maximum cophenetic distance for all
        clusters
    Returns:
        clss, a numpy.ndarray of shape (n,) containing
        the cluster indices for each data point
    """
    Z = scipy.cluster.hierarchy.linkage(X,
                                        method='ward')

    fig = plt.figure(figsize=(25, 10))
    dn = scipy.cluster.hierarchy.dendrogram(Z,
                                            color_threshold=dist)
    plt.show()

    return scipy.cluster.hierarchy.fcluster(Z,
                                            t=dist,
                                            criterion='distance')
