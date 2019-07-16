# -*- coding: utf-8 -*-
'''
Creat on 2019-07-15

Authors: shengzhixu

Email: 

'''

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


def _sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    if len(x) != len(s):
        raise Exception

    # Find the period
    T = s[1] - s[0]

    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM / T))
    return y


def Keystone(data, omega_d, omega_rd, verbose=True, *args, **kwargs):

    m, k = data.shape
    M = np.arange(m)
    key_data = np.zeros_like(data)
    if verbose:
        print("Keystone interpolation started........")
    for i in tqdm(range(k)) if verbose else range(k):
        key_data[:, i] = _sinc_interp(data[:, i], M, M*(omega_d/(omega_rd*i + omega_d)))
    if verbose:
        print("Keystone interpolation finished........")

    return key_data


if __name__ == '__main__':
    # Test
    from numpy.fft import fft2, fftshift
    from numpy import log10

    fd = 0.156
    fr = 0.1334
    fdr = 4e-5

    vm = 1/fd
    k = 200
    m = 156

    v = 8
    R = 10

    [M, K] = np.meshgrid(np.arange(m), np.arange(k))

    data = np.exp(2j*np.pi*(fd*v*M + fr*R*K + fdr*v*M*K))
    plt.jet()
    plt.figure()
    plt.imshow(log10(abs(fftshift(fft2(data, [1024, 1024])))))

    ambiguity = 1
    data = data * np.exp(-2j*np.pi*fdr*M*K* (ambiguity*vm))
    plt.figure()
    plt.imshow(log10(abs(fftshift(fft2(data, [1024, 1024])))))
    kdata = Keystone(data, fd, fdr)
    plt.figure()
    plt.imshow(log10(abs(fftshift(fft2(kdata, [1024, 1024])))))



















