#!/usr/bin/env python3
"""Convolutions and Pooling script"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Perform a valid grayscale convolution
    Args:
        images: (numpy.ndarray) with shape (m, h, w)
                containing multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: (numpy.ndarray )with shape (kh, kw) containing
                the kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
    Returns:
        (numpy.ndarray )containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    o_put_h = h - kh + 1
    o_put_w = w - kw + 1

    o_put = np.zeros((m, o_put_h, o_put_w))

    img = np.arange(m)
    for x in range(o_put_h):
        for y in range(o_put_w):
            o_put[img, x, y] = (np.sum(images[img, x:kh+x,
                                              y:kw+y] * kernel, axis=(1, 2)))
    return o_put
