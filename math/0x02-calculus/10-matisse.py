#!/usr/bin/env python3
""" calculates the derivative of a polynomial """


def poly_derivative(poly):
    """calculates the derivative of a polynomial
    Args:
        poly (list): polynomial list
    Returns:
        (list) list of coefficients, derivatives of polynomial
    """
    if poly is None or type(poly) is not list:
        return None
    if len(poly) == 0 or type(poly[0]) is not int:
        return None
    if any(poly):
        derivative = [poly[c] * c for c in range(1, len(poly))]
        if derivative == []:
            return [0]
        return derivative
