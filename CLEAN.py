#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 08:46:05 2019

@author: shengzhixu
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import exp, pi, fft2, fftshift
import time


'''
CLEAN.py Algorithm
'''

class CLEAN(object):
    
    def __init__(self,
                 data,
                 erot=5,
                 zoom=0):
        self.data = data
        self.erot = erot
        self.zoom = zoom
        
    def clean(self):
        M, N = self.data.shape
        [X, Y] = np.meshgrid(np.arange(N), np.arange(M))
        data = self.data
        xvector = np.linspace(-0.5, 0.5, M*2**self.zoom, endpoint=False)
        yvector = np.linspace(-0.5, 0.5, N*2**self.zoom, endpoint=False)
        max_ori = np.max(abs(fftshift(fft2(self.data, [M*2**self.zoom, N*2**self.zoom]))))
        threshold = max_ori / 10**(0.1*self.erot)
        Indx = []
        Indy = []
        tic = time.time()
        # print("CLEAN Algorithm started...")
        while 1:
            spectrum = (fftshift(fft2(data, [M*2**self.zoom, N*2**self.zoom])))
            max_spectrum = np.max(abs(spectrum))
            # print(max_spectrum)
            if max_spectrum < threshold:
                break
            indx, indy = np.argwhere( abs(spectrum) == max_spectrum)[0]
            fy = xvector[indx]
            fx = yvector[indy]
            Indx.append(fx)
            Indy.append(fy)
            data = data - spectrum[indx, indy] * exp(2j*pi*(fx*X + fy*Y))/M/N
        # print("CLEAN Algorithm ended...")
        # print("Time consumptions: {:.2f} seconds".format(time.time() - tic))

        return Indx, Indy, data


    def thresholding(self):
        # //TUDO add threshold method to accelerate the algoritm

        M, N = self.data.shape
        [X, Y] = np.meshgrid(np.arange(N), np.arange(M))
        data = self.data
        xvector = np.linspace(-0.5, 0.5, M*2**self.zoom, endpoint=False)
        yvector = np.linspace(-0.5, 0.5, N*2**self.zoom, endpoint=False)
        max_ori = np.max(abs(fftshift(fft2(self.data, [M*2**self.zoom, N*2**self.zoom]))))
        threshold = max_ori / 10**(0.1*self.erot)
        Indx = []
        Indy = []


#%%
if __name__ == "__main__":
    M, N = 16, 32
    f = np.array([[0.1, 0.2], [0.3, -0.2]])
    data = np.zeros((M, N), dtype=complex)
    [X, Y] = np.meshgrid(np.arange(N), np.arange(M))
    for i in range(f.shape[0]):
        data = data + exp(2j*pi*(f[i, 0]*X + f[i, 1]*Y))

    Ix, Iy = CLEAN(data, zoom=2).clean()
    print(Ix)
    print(Iy)





















