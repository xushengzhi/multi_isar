# -*- coding: utf-8 -*-
'''
Creat on 2019-02-28

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''


# from __future__ import division
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from pylab import fft,fftshift, pi, exp
import tqdm


def frft(f, a):
    """
    Calculate the fast fractional fourier transform.
    Parameters
    ----------
    f : np array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : np array
        The transformed signal.
    References
    ---------
     .. [1] This algorithm implements `frft.m` from
        https://nalag.cs.kuleuven.be/research/software/FRFT/
    """
    ret = np.zeros_like(f, dtype=np.complex)
    f = f.copy().astype(np.complex)
    N = len(f)
    shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
    sN = np.sqrt(N)
    a = np.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        return f
    if a == 2.0:
        return np.flipud(f)
    if a == 1.0:
        ret[shft] = np.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = np.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2.0
        f = np.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = np.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha / 2)
    sina = np.sin(alpha)
    f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T

    # chirp premultiplication
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
                     np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = np.pi / N / sina / 4
    ret = scipy.signal.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def ifrft(f, a):
    """
    Calculate the inverse fast fractional fourier transform.
    Parameters
    ----------
    f : np array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : np array
        The transformed signal.
    """
    return frft(f, -a)


def sincinterp(x):
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]

if __name__ == "__main__":
    size = 128
    # [X, Y] = np.meshgrid(np.arange(size), np.arange(size))
    # X = np.exp(2j*np.pi*(0.1*X + 0.2*Y + 0.01*Y*Y))
    # X = fftshift(fft(X, axis=-1))
    # Dat = np.zeros((size, 100, size), dtype=complex)
    # for i in tqdm.tqdm(range(size)):
    #     for j, alpha in enumerate(np.linspace(0, 4, 100, endpoint=False)):
    #         Dat[i, j, :] = frft(X[i, :], alpha)
    #
    # plt.imshow(abs(Dat[0, :, :]), aspect='auto', cmap='jet', extent=[-0.5, 0.5, -2, 2])
    #
    # dat = Dat[0, :, :].flatten()
    # indx = np.argwhere(abs(dat) == max(abs(dat)))
    # np.arctan(dat[indx])

    x = np.arange(size)
    y = exp(2j*pi*( 0.1*x + 0.000*x*x ))
    Dat = np.zeros((100, size), dtype=complex)
    for j, alpha in enumerate(np.linspace(0, 4, 100, endpoint=False)):
        Dat[j, :] = frft(y, alpha)

    plt.imshow(np.log10(abs(Dat)),
               extent=[-0.5, 0.5, 0, 4 ],
               aspect='auto',
               cmap='jet')