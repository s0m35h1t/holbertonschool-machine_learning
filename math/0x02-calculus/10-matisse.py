#!/usr/bin/env python3
""" calculates the derivative of a polynomial """


def poly_derivative(poly):
    """calculates the derivative of a polynomial
    Args:
        poly (list): polynomial list
    Returns:
        (list) list of coefficients, derivatives of polynomial
    """
    if not type(poly) is list:
        return None
    if len(poly) == 0:
        return None
    if type(poly[0]) is not int:
        return None
    derivative = [poly[c] * c for c in range(1, len(poly))]
    if derivative == []:
        derivative = [0]
    return derivative
