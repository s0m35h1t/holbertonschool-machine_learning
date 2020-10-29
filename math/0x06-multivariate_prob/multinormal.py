#!/usr/bin/env python3
"""Define MultiNormal Class"""

import numpy as np


class MultiNormal():
    """Class multinormal"""

    def __init__(self, data):
        """
        Init  Multinormal Class
        Args:
            data is a numpy.ndarray of shape (d, n) containing the data set:
                n is the number of data points
                d is the number of dimensions in each data pointa
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = (np.mean(data, axis=1)).reshape(d, 1)

        X = data - self.mean
        self.cov = ((np.matmul(X, X.T)) / (n - 1))

    def pdf(self, x):
        """Calculates the PDF at a data point
        Args:
            x (np.ndarray): matrix of shape (d, 1) containing the data point
                            whose PDF should be calculated.
        Returns:
            the value of the
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        ds = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != ds:
            raise ValueError("x must have the shape ({}, 1)".format(ds))

        covdet = np.linalg.det(self.cov)
        covinv = np.linalg.inv(self.cov)
        x_minus_u = x - self.mean

        pdf_1 = 1 / np.sqrt(((2 * np.pi) ** ds) * covdet)
        pdf_2 = np.exp(np.matmul(np.matmul(-x_minus_u.T / 2, covinv),
                                 x_minus_u))

        pdf = pdf_1 * pdf_2

        return pdf.flatten()[0]
