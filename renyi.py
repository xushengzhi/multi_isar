# -*- coding: utf-8 -*-
'''
Creat on 2019-04-23

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from pylab import exp, pi, fft2, fftshift
from multi_isar.utils import renyi, normalize

# TODO: Algorithm

'''
A:  Range alignment during observation time
A1: peak tracking
A2: envelope correction
A3: global range alignment

B:  Optimization of a focusing indicator
B1: Doppler centroid tracking
B2: prominent point processing
B3: phase gradient autofocus
B4: Entropy minimization
B5: peak value maximization
B6: contrast maximization

'''


#############################################################################
# image contrast method
def spatial_mean(image):
    energy = np.sum(abs(image))
    return energy/image.size

def imgae_constrast(image):

    spatial_mean_image = spatial_mean(image)
    return np.sqrt(spatial_mean((image - spatial_mean_image)**2)) / spatial_mean_image
#############################################################################


# To test the performance of ZZ^H and Z^HZ

X, Y = np.meshgrid(np.arange(128), np.arange(32))
Z = exp(2j*pi*(0.1*X + 0.1*Y + 0.001*X*Y)) + exp(2j*pi*(0.3*X - 0.1*Y - 0.002*X*Y))

IC = []
IE = []
EE1 = []
EE2 = []
v_scan = np.linspace(-0.005, 0.005, 200)
for k in tqdm(v_scan):
    C = exp(2j*pi*k*X*Y)
    Zcom = Z * C.conj()
    EE1.append(renyi(np.linalg.eigvalsh(Zcom.dot(Zcom.T.conj()))))
    EE2.append(renyi((np.linalg.eigvalsh(Zcom.T.conj().dot(Zcom)))))
    # Zfft = abs(fft2(Zcom))
    # IC.append(imgae_constrast(Zfft))
    # IE.append(renyi(Zfft, alpha=0.5))

# plt.plot(v_scan, normalize(IC))
# plt.plot(v_scan, normalize(IE))
plt.plot(v_scan, normalize(EE1), label='EE1')
plt.plot(v_scan, normalize(EE2), label='EE2', ls=':')
plt.legend()










