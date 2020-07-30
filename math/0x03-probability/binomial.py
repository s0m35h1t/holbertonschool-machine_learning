#!/usr/bin/env python3
""" Binomial distribution class """


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


class Binomial:
    """ binomial dist class """

    def __init__(self, data=None, n=1, p=0.5):
        """ initialize constructor for binomial class """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            m = sum(data) / len(data)
            variance = sum([(x - m) ** 2 for x in data]) / len(data)
            self.p = -1 * (variance / m - 1)
            n = m/self.p
            self.n = round(n)
            self.p *= n/self.n

    def pmf(self, k):
        """ calculates PMF for a given number of successes
        Args:
            k (int):s the number of “successes”
        Returns:
            (float) the PMF value for k
        """

        if int(k) > self.n or k < 0:
            return 0
        return (fact(self.n) / fact(int(k)) / fact(self.n - int(k)) *
                self.p ** int(k) * (1 - self.p) ** (self.n - int(k)))

    def cdf(self, k):
        """ calculates value of CDF for given number of successes
        Args:
            k (int):s the number of “successes”
        Returns:
            (float) the PMF value for k
        """
        if k > self.n or k < 0:
            return 0
        sum = 0
        for i in range(int(k) + 1):
            sum += self.pmf(i)
        return sum
