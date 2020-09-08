#!/usr/bin/env python3
"""Convolutions and Pooling script"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Perform a graysclae convolution
    Args:
        images: (numpy.ndarray )with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: (numpy.ndarray )with shape (kh, kw) containing
                the kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
        padding: tuple of (ph, pw)
                 ph: is the padding for the height of the image
                 pw: is the padding for the width of the image
    Returns:
        (numpy.ndarray )containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    pad_w = padding[1]
    pad_h = padding[0]
    o_put_h = h + (2 * pad_h) - kh + 1
    o_put_w = w + (2 * pad_w) - kw + 1

    img_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                        (pad_w, pad_w)), mode='constant')

    o_put = np.zeros((m, o_put_h, o_put_w))

    img = np.arange(m)
    for x in range(o_put_h):
        for y in range(o_put_w):
            o_put[img, x, y] = (np.sum(img_pad[img,
                                               x:kh+x, y:kw+y] * kernel,
                                       axis=(1, 2)))
    return o_put
