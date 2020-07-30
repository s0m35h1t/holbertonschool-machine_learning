#!/usr/bin/env python3
"""Define: Normal class"""


class Normal:
    """Normal class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """ initialize constructor for normal dist """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum([(x - self.mean) ** 2 for x in data])
                                 / (len(data))) ** .5)

    def z_score(self, x):
        """ calculates z score for x value
        Args:
            x (int): the x-value
        Returns
            Returns the x-value of z
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates x value at that z score
        Args:
            x (int): the x-value
        Returns
            the PDF value for x
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculates pdf at given x value
        Args:
            x (int): the x-value
        Returns
            (float) the CDF value for x
        """
        return (pow(2.7182818285, ((x - self.mean) ** 2 /
                                   (-2 * self.stddev ** 2))) /
                (2 * 3.1415926536 * self.stddev ** 2) ** .5)

    def cdf(self, x):
        """ calculates cdf at given x value
        Args:
            x (int); the x-value
        Returns
            (float)
        """
        e = (x - self.mean) / (self.stddev * 2 ** .5)
        return (1 + (e - e ** 3 / 3 + e ** 5 / 10 - e ** 7
                     / 42 + e ** 9 / 216) * 2 / 3.1415926536 ** .5) / 2
