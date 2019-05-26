# -*- coding: utf-8 -*-
'''
Creat on 2019-04-09

Authors: shengzhixu

Email: 

'''

'''
TEST eigen spectrum of AA^H and A^HA
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import pi, exp, fft2, fftshift, log10

from utils import renyi, entropy


def normalize(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def db(mag):
    return 20*np.log10(abs(mag+1e-8))

def entropy(vector):
    info = vector/np.sum(vector)
    return -sum(info * log10(abs(info)))

def awgn(sig, snrdb, sigpower=0):
    """
    Additive white gaussian noise.  Assumes signal power is 0 dBW
    """
    L = sig.size
    sigpower = 10*np.log10(np.sum(np.abs(sig)**2)/L)

    if sp.iscomplexobj(sig):
        noise = (sp.randn(*sig.shape) + 1j*sp.randn(*sig.shape))/np.sqrt(2)
    else:
        noise = sp.randn(*sig.shape)
    noisev = 10**((sigpower - snrdb)/20)
    return sig + noise*noisev

xsize = 16
ysize = 16
X, Y = np.meshgrid(np.arange(xsize), np.arange(ysize))

numtar= 100
fx = np.linspace(-0.3, 0.3, numtar)
fy = np.ones((numtar, ))*0

Z= np.zeros_like(X, dtype=complex)
for i in range(numtar):
    Z = Z + exp(2j*pi*( fx[i]*X + fy[i]*Y ))

Zfft = fftshift(fft2(awgn(Z, 10), [512, 512]))
plt.imshow(db(Zfft), cmap='jet', aspect='auto')

b = entropy(np.linalg.eigvalsh(Z.dot(Z.T.conj())))
a = entropy(np.linalg.eigvalsh(Z.T.conj().dot(Z)))



c = np.linspace(-0.01, 0.01, 50)
B = []
A = []
for i in c:
    C = Z * exp(2j*pi*i*X*Y)
    B.append(entropy(np.linalg.eigvalsh(C.dot(C.T.conj()))))
    A.append(entropy(np.linalg.eigvalsh(C.T.conj().dot(C))))

plt.figure()
plt.plot(c, B, label='B', lw=4, color='r')
plt.plot(c, A, label='A', ls=':', lw=1.5, color='b')
plt.legend()




