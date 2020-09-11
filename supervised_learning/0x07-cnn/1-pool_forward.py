#!/usr/bin/env python3
"""Convolutional Neural Networks"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Forward propagate over a pooling layer in a NN
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
                m: is the number of examples
                h_prev: the height of the previous layer
                w_prev: the width of the previous layer
                c_prev: the number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
                      the pooling
                      kh: the kernel height
                      kw: the kernel width
        stride: tuple of (sh, sw) containing the strides for the convolution
                sh: the stride for the height
                sw: the stride for the width
        mode: string containing either max or avg, indicating whether to
              perform maximum or avg pooling, respectively
    Returns:
        the output of the pooling layer
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    sh, sw = stride
    c_h = int((h_prev - kh) / sh) + 1
    c_w = int((w_prev - kw) / sw) + 1

    conv = np.zeros((m, c_h, c_w, c_prev))

    for x in range(c_h):
        for y in range(c_w):
            if mode == 'max':
                conv[:, x, y] = (np.max(A_prev[:,
                                               x * sh:((x * sh) + kh),
                                               y * sw:((y * sw) + kw)],
                                        axis=(1, 2)))
            elif mode == 'avg':
                conv[:, x, y] = (np.mean(A_prev[:,
                                                x * sh:((x * sh) + kh),
                                                y * sw:((y * sw) + kw)],
                                         axis=(1, 2)))
    return conv
