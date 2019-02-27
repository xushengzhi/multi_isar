#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:58:07 2019

@author: shengzhixu
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pylab import pi, exp, fft, fft2, log10, fftshift, sin, cos, deg2rad
from scipy.constants import speed_of_light as c
from scipy.io import loadmat

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


# %% settings of simulation
save_fig = False


# %% setting basic parameters
n = 128  # slow time
m = 256  # fast time
SNR = -5
[X, Y] = np.meshgrid(np.arange(n), np.arange(m))  # fast time, slow time

fight = loadmat('/Users/shengzhixu/Documents/Python/multi_isar/Fighter3.mat')
Xc = fight['Xc'].flatten()/10
Yc = fight['Yc'].flatten()/10

plt.figure()
plt.scatter(Xc, Yc)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
if save_fig:
    plt.savefig("Target.png", dpi=300)

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

fdr = mu*2*T/c/fs

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

def fold(value):
    return value - np.floor(value + 0.5)

round_range1 = fold((R[0] + Yc)*fr)
round_range2 = fold((R[1] + Yc)*fr)
round_range3 = fold((R[2] + Yc)*fr)
round_velocity1 = fold((v[0] + w[0]*Xc)*fd)
round_velocity2 = fold((v[1] + w[1]*Xc)*fd)
round_velocity3 = fold((v[2] + w[2]*Xc)*fd)


low_alpha = 0.5         # variation of the amplitude (real)
for i in range(number_target):
    data1 = data1 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[0]+Yc[i]) * X +
                                 fd*(v[0]+w[0]*Xc[i]) * Y +
                                 c1*X*Y))

for i in range(number_target):
    data2 = data2 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[1]+Yc[i])* X +
                                 fd*(v[1]+w[1]*Xc[i]) * Y +
                                 c2*X*Y))

for i in range(number_target):
    data3 = data3 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j * pi * (fr* (R[2] + Yc[i]) * X +
                                    fd* (v[2] + w[2] * Xc[i]) * Y +
                                    c3 * X * Y))

data = awgn(data1 + data2 + data3, SNR)
dataf = fftshift(fft((data)* exp(-2j*pi*0.*X*Y), axis=-1, n=4*m), axes=-1)
plt.figure(figsize=[12, 8])
plt.imshow(20*log10(abs(dataf)), aspect='auto', cmap='jet')
plt.clim(vmin=22, vmax=62)
plt.colorbar()
if save_fig:
    plt.savefig("1DFFT.png", dpi=300)

dataf = fftshift(fft2((data)* exp(-2j*pi*0.*X*Y), [4*n, 4*m]))
plt.figure(figsize=[12, 8])
plt.imshow(20*log10(abs(dataf)), aspect='auto', cmap='jet')
plt.clim(vmin=50, vmax=90)
plt.colorbar()
if save_fig:
    plt.savefig("2DFFT.png", dpi=300)


# %% Using ME or VSVD to separate targets and estimate the couplings
cle = 200                                   # number of the searching grids
vspan = np.linspace(5, 15, cle)
cspan = vspan * fdr


def minimum_entropy(cspan, data, X, Y, fdr):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        isar = abs(fft2(datac, [4*m, 4*n]))
        info = isar/np.sum(isar)
        emat = info * log10(info)
        ec[i] = -np.sum(emat)
    # print("Time for ME: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(ec) == ec).flatten()[0]
    cvalue = cspan[indc]
    print("The estimate vr by ME is {:.3f}.".format(cvalue/fdr))

    return ec, cvalue


def varsvd(cspan, data, X, Y, fdr):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        es[i] = -np.var(np.linalg.svd(datac, compute_uv=False))
    # print("Time for SVD: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]
    print("The estimate vr by SVD is {:.3f}.".format(cvalue/fdr))

    return es, cvalue


ec, _ = minimum_entropy(cspan, data, X, Y, fdr)
es, com = varsvd(cspan, data, X, Y, fdr)

plt.figure(figsize=[12, 8])
plt.plot(vspan, normalize(ec), label="Entropy", lw=2, color='g')
plt.plot(vspan, normalize(es), label="Variance of SVD", lw=2, color='r')
plt.vlines(np.array(vr), ymin = -1.5, ymax=0.5, linestyle='--', color='b')
plt.legend(loc='lower left')
plt.xlabel("Velocity (m/s)")
if save_fig:
    plt.savefig("MEvsVSVD.png", dpi=300)


# %% CLEAN technique
method = varsvd  #
new_data = data.copy()
_, new_com = method(cspan, data, X, Y, fdr)

indx1, indy1, new_data = CLEAN(data = new_data * exp(2j*pi*new_com*X*Y), zoom=2, erot=3).clean()
new_data = new_data * exp(-2j*pi*new_com*X*Y)

_, new_com = method(cspan, new_data, X, Y, fdr)
indx2, indy2, new_data = CLEAN(data = new_data * exp(2j*pi*new_com*X*Y), zoom=2, erot=3).clean()
new_data = new_data * exp(-2j*pi*new_com*X*Y)

plt.figure()
plt.imshow(20*log10(abs(fft2(new_data * exp(2j*pi*new_com*X*Y)))), aspect='auto', cmap='jet')
plt.colorbar()

_, new_com = method(cspan, new_data, X, Y, fdr)
indx3, indy3, new_data = CLEAN(data = new_data * exp(2j*pi*new_com*X*Y), zoom=2, erot=3).clean()




# compare SE
plt.figure(figsize=[12, 8])
plt.scatter(indx1, indy1, marker='o', s=60, label="Target1")
plt.scatter(indx2, indy2, marker='o', s=60, label="Target1")
plt.scatter(indx3, indy3, marker='o', s=60, label="Target1")
plt.scatter(-round_range1, -round_velocity1, marker='x', c='k', label="True Pos")
plt.scatter(-round_range2, -round_velocity2, marker='x', c='k')
plt.scatter(-round_range3, -round_velocity3, marker='x', c='k')
plt.legend(loc="upper center")
plt.xlabel("X")
plt.ylabel("Y")
if save_fig:
    plt.savefig("ME.png", dpi=300)




