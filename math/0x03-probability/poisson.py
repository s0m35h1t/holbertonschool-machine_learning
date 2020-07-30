#!/usr/bin/env python3
""" Define: Class Poisson represents poisson distribution """


def fact(n):
    """ returns the factorial of n
    Args:
        n (int): integer
    Returns:
        (int) n factorial
    """
    if n < 0:
        return None
    if n == 0:
        return 1
    if n < 2:
        return 1
    return n * fact(n-1)


class Poisson:
    """ Poisson distribution class """

    def __init__(self, data=None, lambtha=1.):
        """Poisson intialization"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """calculates value of PMF for given number of successes
        Args:
            k (int): the number of “successes”
        Returns:
            (int) the PMF value for k
        """
        if k < 0:
            return 0
        return (pow(self.lambtha, int(k)) *
                pow(2.7182818285, -1 * self.lambtha) /
                fact(int(k)))

    def cdf(self, k):
        """ Calculates value of the CDF for a given number of successes
        Args:
            k (int): the number of “successes” )
        Retuns:
            (int)  the CDF value for k
        """
        if k < 0:
            return 0
        return sum([self.pmf(n) for n in range(int(k) + 1)])
