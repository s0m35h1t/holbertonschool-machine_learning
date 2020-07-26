
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
    if len(poly) == 0 or type(poly[0]) is not int:
        return None
    _unused, *poly = poly
    if any(poly):
        derivative = [p * c for p, c in enumerate(poly, 1)]
        if derivative == 0:
            return [0]
        return derivative
