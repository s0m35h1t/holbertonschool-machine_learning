#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """calculates integral of a poly
    Args
        poly (list): polynomial list
        C (int): representing the integration constant
    Returns
        (list): list of coefficients
    """
    if poly is None or not all(type(C) in (float, int) for c in poly):
        return None
    if type(C) is not int:
        return
    integral = [coeff/a if coeff % a != 0 else coeff //
                a for a, coeff in enumerate(poly, 1)]
    while len(integral) > 0 and integral[-1] == 0:
        integral.pop()
    return [C] + integral
