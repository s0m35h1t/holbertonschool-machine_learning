#!/usr/bin/env python3
"""Define: Exponential class """


class Exponential:
    """ Exponential class """

    def __init__(self, data=None, lambtha=1.):
        """ initial class constructor """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """ pdf of exp distribution of x
        Agrs:
            x (int): the time period
        Returns:
            (int) the PDF value for x
        """
        if x < 0:
            return 0
        return self.lambtha * pow(2.7182818285, -1 *
                                  self.lambtha * x)

    def cdf(self, x):
        """ cdf of exp distribution of x
        Args:
            x (int): the time period
        Returns:
            (int) the CDF value for x
        """
        if x < 0:
            return 0
        return 1 - pow(2.7182818285, -1 * self.lambtha * x)
