#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:58:07 2019

@author: shengzhixu
"""
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import pi, exp, fft, fft2, log10, fftshift, sin, cos, deg2rad
from scipy.io import loadmat
from scipy.constants import speed_of_light as c

from multi_isar.CLEAN import CLEAN

'''
Mutli-targets ISAR separation comparison between

1) Minimum Entropy
2) Variance of the Sigular values

Conclusions:
    1) Variance has better separation performance
    2) Variance cannot resolve higher order couplings
        for example:
            if the coupling has X*Y^2 and X*Y, they cannot be resolved correctly
            by svd as they located in the same sigular vector
'''

#os.chdir('~/Documents/Python')
plt.close('all')

# %% Useful functions
def normalize(data):
    return (data - np.max(data))/(np.max(data) - np.min(data))


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


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def db(mag):
    return 20*np.log10(abs(mag))


def mplot(pic, clim=40, **kwargs):
    vmax = np.max(pic)
    plt.imshow(pic, aspect='auto', cmap='gray_r')
    plt.colorbar()
    plt.clim([vmax-clim, vmax])

# %% setting basic parameters
n = 256  # slow time
m = 256  # fast time
SNR = -5
[X, Y] = np.meshgrid(np.arange(n), np.arange(m))  # fast time, slow time

fight = loadmat('/Users/shengzhixu/Documents/Python/multi_isar/Fighter3.mat')
Xc = fight['Xc'].flatten()/10
Yc = fight['Yc'].flatten()/10

plt.figure()
plt.scatter(Xc, Yc)

number_target = Xc.size

# c = - \mu * 2 * v * T / c / fs

# %%
B = 4e9
resolution = c/2/B
fc = 78e9
T = 1e-3
Td = 0.8e-3

Ts = T/m
fs = 1/Ts
mu = B/Td
vm = c/4/T/fc

fdr = -mu*2*T/c/fs

# %% Targets parameters
R = [11.678, 12.345, 13.123]        # inital range
v = [10, 9, 11]         # velocity ()
a = [0, 0, 0]           # acceleration
theta = [30, 31, 32]    # angle
w = [0, 0, 0]           # rotatioal velocity
vr = v*cos(deg2rad(theta))  # radial velocity
print(vr)
vt = v*sin(deg2rad(theta))  # translational velocity

c1, c2, c3 = fdr * vr
w = w + vt/R            # rotatioal velocity + translational_velocity / range

# %% Generating data
data1 = np.zeros((m, n), dtype=complex)
data2 = np.zeros_like(data1, dtype=complex)
data3 = np.zeros_like(data1, dtype=complex)

fr = mu*2/c/fs
fd = fc*T*2/c

round_range1 = (R[0] + Yc)*fr
round_range2 = (R[1] + Yc)*fr
round_range3 = (R[2] + Yc)*fr
round_range1 = round_range1 - np.floor(round_range1 + 0.5)
round_range2 = round_range2 - np.floor(round_range2 + 0.5)
round_range3 = round_range3 - np.floor(round_range3 + 0.5)
round_velocity1 = (v[0] + w[0]*Xc)*fd
round_velocity2 = (v[1] + w[1]*Xc)*fd
round_velocity3 = (v[2] + w[2]*Xc)*fd
round_velocity1 = round_velocity1 - np.floor(round_velocity1 + 0.5)
round_velocity2 = round_velocity2 - np.floor(round_velocity2 + 0.5)
round_velocity3 = round_velocity3 - np.floor(round_velocity3 + 0.5)

for i in range(number_target):
    data1 = data1 + exp(-2j*pi*( fr*(R[0]+Yc[i]) * X +
                                 fd*(v[0]+w[0]*Xc[i]) * Y +
                                 c1*X*Y))

for i in range(number_target):
    data2 = data2 + exp(-2j*pi*( fr*(R[1]+Yc[i])* X +
                                 fd*(v[1]+w[1]*Xc[i]) * Y +
                                 c2*X*Y))

for i in range(number_target):
    data3 = data3 + exp(-2j * pi * (fr* (R[2] + Yc[i]) * X +
                                    fd* (v[2] + w[2] * Xc[i]) * Y +
                                    c3 * X * Y))

data = awgn(data1 + data2 + data3, SNR)
dataf = fftshift(fft((data)* exp(-2j*pi*0.*X*Y), axis=-1, n=4*m), axes=-1)
dataf = fftshift(fft2((data)* exp(-2j*pi*0.*X*Y), [4*n, 4*m]))
plt.figure()
plt.imshow(log10(abs(dataf)), aspect='auto', cmap='jet')
plt.colorbar()


# %% Using ME or VSVD to separate targets and estimate the couplings
cle = 200                                   # number of the searching grids
vspan = np.linspace(-20, 20, cle)
cspan = vspan * fdr

def minimum_entropy(cspan, data, X, Y, fdr):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(-2j*pi*com*X*Y)
        isar = abs(fft2(datac, [2*m, 2*n]))
        info = isar/np.sum(isar)
        emat = info * log10(info)
        ec[i] = -np.sum(emat)
    # print("Time for ME: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(ec) == ec).flatten()[0]
    cvalue = cspan[indc]
    print("The estimate vr by ME is {:.3f}.".format(-cvalue/fdr))

    return ec, cvalue

ec, _ = minimum_entropy(cspan, data, X, Y, fdr)

def varsvd(cspan, data, X, Y, fdr):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(-2j*pi*com*X*Y)
        es[i] = -np.sum(np.var(np.linalg.svd(datac, compute_uv=False)))
    # print("Time for SVD: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]
    print("The estimate vr by SVD is {:.3f}.".format(-cvalue/fdr))

    return es, cvalue

es, com = varsvd(cspan, data, X, Y, fdr)

plt.figure()
plt.plot(vspan, normalize(ec), label="Minimum Entropy", lw=2, color='g')
plt.plot(vspan, normalize(es), label="Variance of SVD", lw=2, color='r')
plt.vlines(-np.array(vr), ymin = -2, ymax=1, linestyle='--', color='b')
plt.legend(loc='lower left')


# %% CLEAN technique
method = varsvd  #
new_data = data.copy()
_, new_com = method(cspan, data, X, Y, fdr)

indx1, indy1, new_data = CLEAN(data = new_data * exp(-2j*pi*new_com*X*Y), zoom=2, erot=3).clean()
plt.figure()
plt.scatter(indx1, indy1)
new_data = new_data * exp(2j*pi*new_com*X*Y)

new_es, new_com = method(cspan, new_data, X, Y, fdr)
indx2, indy2, new_data = CLEAN(data = new_data * exp(-2j*pi*new_com*X*Y), zoom=2, erot=3).clean()
plt.figure()
plt.scatter(indx2, indy2)
new_data = new_data * exp(2j*pi*new_com*X*Y)

new_es, new_com = method(cspan, new_data, X, Y, fdr)
indx3, indy3, new_data = CLEAN(data = new_data * exp(-2j*pi*new_com*X*Y), zoom=2, erot=3).clean()
plt.figure()
plt.scatter(indx3, indy3)


# compare SE

plt.figure(figsize=[12, 8])
plt.scatter(indx1, indy1, marker='o', s=60)
plt.scatter(indx2, indy2, marker='o', s=60)
plt.scatter(indx3, indy3, marker='o', s=60)
plt.scatter(-round_range1, -round_velocity1, marker='x', c='w')
plt.scatter(-round_range2, -round_velocity2, marker='x', c='w')
plt.scatter(-round_range3, -round_velocity3, marker='x', c='w')




