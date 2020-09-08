#!/usr/bin/env python3
"""Convolutions and Pooling script"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform a grayscale same convolution
    Args:
        images: (numpy.ndarray )with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
        kernel: (numpy.ndarray )with shape (kh, kw) containing the
                kernel for the convolution
                kn: the height of the kernel
                kw: the width of the kernel
    Returns:
        (numpy.ndarray) containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    pad_h = int((kh - 1) / 2)
    pad_w = int((kw - 1) / 2)

    if kh % 2 == 0:
        pad_h = int(kh / 2)
    if kw % 2 == 0:
        pad_w = int(kw / 2)

    img_pad = np.pad(images, pad_width=((0, 0),
                                        (pad_h, pad_h), (pad_w, pad_w)),
                     mode='constant')

    o_put = np.zeros((m, h, w))

    img = np.arange(m)
    for x in range(h):
        for y in range(w):
            o_put[img, x, y] = (np.sum(img_pad[img,
                                               x:kh+x, y:kw+y] * kernel,
                                       axis=(1, 2)))
    return o_put
