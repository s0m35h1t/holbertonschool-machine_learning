#!/usr/bin/env python3
"""Convolutions and Pooling script"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a grayscale convolution
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
        padding: is either a tuple of (ph, pw), 'same, 'valid'
                 ph: is the padding for the height of the image
                 pw: is the padding for the width of the image
        stride: is a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
    Returns:
        (numpy.ndarray )containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    pad_w = 0
    pad_h = 0
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        pad_h = int((((h - 1) * sh + kh - h) / 2) + 1)
        pad_w = int((((w - 1) * sw + kw - w) / 2) + 1)
    if type(padding) == tuple:
        pad_h = padding[0]
        pad_w = padding[1]

    img_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                        (pad_w, pad_w)), mode='constant')

    o_put_h = int(((h + 2 * pad_h - kh) / sh) + 1)
    o_put_w = int(((w + 2 * pad_w - kh) / sw) + 1)

    o_put = np.zeros((m, o_put_h, o_put_w))

    img = np.arange(m)
    for x in range(o_put_h):
        for y in range(o_put_w):
            o_put[img, x, y] = (np.sum(img_pad[img,
                                               x * sh:((x * sh) + kh),
                                               y * sw:((y * sw) + kw)] * kernel,
                                       axis=(1, 2)))
    return o_put
