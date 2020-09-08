#!/usr/bin/env python3
"""Convolutions and Pooling script"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform a convolution of img with channels
    Args:
        images: (numpy.ndarray) with shape (m, h, w) containing
                multiple grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
                c: number the channels in the image
        kernel_shape: tuple of (kh, kw) containing
                the kernel shape of the pooling
                kh: the height of the kernel
                kw: the width of the kernel
        stride: is a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
        mode: indicates the type of pooling
                 max: max pooling
                 avg: average pooling
        Returns:
            (numpy.ndarray) containing the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    o_put_h = int(1 + ((h - kh) / sh))
    o_put_w = int(1 + ((w - kw) / sw))

    o_put = np.zeros((m, o_put_h, o_put_w, c))

    img = np.arange(m)
    for x in range(o_put_h):
        for y in range(o_put_w):
            if mode == 'max':
                o_put[img, x, y] = (np.max(images[img,
                                                  x * sh:((x * sh) + kh),
                                                  y * sw:((y * sw) + kw)],
                                           axis=(1, 2)))
            elif mode == 'avg':
                o_put[img, x, y] = (np.mean(images[img,
                                                   x * sh:((x * sh) + kh),
                                                   y * sw:((y * sw) + kw)],
                                            axis=(1, 2)))
    return o_put
