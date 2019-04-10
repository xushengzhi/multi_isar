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
from skimage.transform import hough_line
from tqdm import tqdm

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
    return (data - np.min(data))/(np.max(data) - np.min(data))


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
m = 128  # fast time
SNR = 0
[X, Y] = np.meshgrid(np.arange(n), np.arange(m))  # fast time, slow time
model_zoom = 1

fight = loadmat('/Users/shengzhixu/Documents/Python/multi_isar/Fighter2.mat')
Xc = fight['Xc'].flatten()
Yc = fight['Yc'].flatten()
Xc = Xc / np.max(Xc) * 1.5 /model_zoom
Yc = Yc / np.max(Yc) * 1.5 /model_zoom

plt.figure()
plt.scatter(Xc, Yc)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
if save_fig:
    plt.savefig("Target.png", dpi=300)

number_scatters = Xc.size
number_target = 3
# c = - \mu * 2 * v * T / c / fs
# %%
B = 4e9     # bandwidth
resolution = c/2/B
fc = 78e9   # carrier frequency
T = 1e-3
Td = 0.8e-3

Ts = Td/m
fs = 1/Ts
mu = B/Td
vm = c/4/T/fc



# %% Targets parameters
R = [10.678, 10.677, 12.823]        # inital range
v = np.array([-9, 10, 11]) * 1        # velocity ()
a = np.array([20, 15, 25]) * 0.3         # acceleration

theta = [30, 31, 32]    # angle
w = [0, 0, 0]           # rotational velocity
vr = v*cos(deg2rad(theta))  # radial velocity
ar = a*cos(deg2rad(theta))
print(vr)
vt = v*sin(deg2rad(theta))  # translational velocity


w = w + vt/R            # rotational velocity + translational_velocity / range

# %% Generating data
data1 = np.zeros((m, n), dtype=complex)
data2 = np.zeros_like(data1, dtype=complex)
data3 = np.zeros_like(data1, dtype=complex)

fr = mu*2/c/fs              # k    * y
fd = fc*T*2/c               # m    * x
frs = fc/c*T**2             # m^2  * a
fdr = mu*2*T/c/fs           # mk   * v
fa = mu/c*T**2/fs * 1       # km^2 * a

c1, c2, c3 = fdr * vr

def fold(value):
    return value - np.floor(value + 0.5)

round_range1 = fold((R[0] + Yc)*fr)
round_range2 = fold((R[1] + Yc)*fr)
round_range3 = fold((R[2] + Yc)*fr)
round_velocity1 = fold((vr[0] + w[0]*Xc)*fd)
round_velocity2 = fold((vr[1] + w[1]*Xc)*fd)
round_velocity3 = fold((vr[2] + w[2]*Xc)*fd)


low_alpha = 0.5        # variation of the amplitude (real)     # X:fast time
for i in range(number_scatters):
    data1 = data1 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[0]+Yc[i]) * X +
                                 fd*(vr[0]+w[0]*Xc[i]) * Y +
                                 ar[0] * fa * X * Y * Y +
                                 ar[0] * frs * Y * Y +
                                 c1 * X * Y))

for i in range(number_scatters):
    data2 = data2 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[1]+Yc[i]) * X +
                                 fd*(vr[1]+w[1]*Xc[i]) * Y +
                                 ar[1] * fa * X * Y * Y +
                                 ar[1] * frs * Y * Y +
                                 c2 * X * Y))

for i in range(number_scatters):
    data3 = data3 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[2]+Yc[i]) * X +
                                 fd*(vr[2]+w[2]*Xc[i]) * Y +
                                 ar[2] * fa * X * Y * Y +
                                 ar[2] * frs * Y * Y +
                                 c3 * X * Y))

if number_target == 3:
    data = data1 + data2 + data3
elif number_target==2:
    data = data1 + data2
else:
    data = data1

data = awgn(data, SNR)
dataf = fftshift(fft((data)* exp(2j*pi*0*X*Y), axis=-1, n=1*m), axes=-1)
plt.figure(figsize=[12, 8])
plt.imshow(20*log10(abs(dataf)), aspect='auto', cmap='jet')
plt.clim(vmin=22, vmax=62)
plt.colorbar()
if save_fig:
    plt.savefig("1DFFT.png", dpi=300)

# # %%Hough Line Transform
# h, theta, d = hough_line(abs(dataf))
# plt.figure()
# plt.imshow(20*log10(abs(h + 1)), aspect='auto', cmap='gray')

# %%
dataf = fftshift(fft2((data)* exp(2j*pi*c1*X*Y), [4*n, 4*m]))
plt.figure(figsize=[12, 8])
plt.imshow(20*log10(abs(dataf)), aspect='auto', cmap='jet')
plt.clim(vmin=50, vmax=90)
plt.colorbar()
if save_fig:
    plt.savefig("2DFFT.png", dpi=300)


# %% Using ME or VSVD to separate targets and estimate the couplings
cle = 200                                   # number of the searching grids for velocity
vspan = np.linspace(-15, 15, cle)
cspan = vspan * fdr


def minimum_entropy(cspan, data, X, Y):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        isar = abs(fft2(datac, [1*n, 1*m]))
        info = isar/np.sum(isar)
        emat = info * log10(info)
        ec[i] = -np.sum(emat)
    print("Time for ME: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(ec) == ec).flatten()[0]
    cvalue = cspan[indc]

    return ec, cvalue


def entropy(vector):
    info = vector/np.sum(vector)
    return -sum(info * log10(abs(info)))


def varsvd(cspan, data, X, Y):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        es[i] = entropy(np.linalg.svd(datac, compute_uv=False))
    print("Time for SVD: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue


def vareig(cspan, data, X, Y):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        # es[i] = entropy((np.linalg.eigvalsh(datac.T.conj().dot(datac))))
        es[i] = entropy((np.linalg.eigvalsh(datac.T.conj().dot(datac))))
    print("Time for EIG: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue

ele = 61                 # number of the searching grids for acceleration
ascan = np.linspace(-1.5*np.max(abs(a)), 1.5*np.max(abs(a)), ele)
# fascan = ascan * fa

def angle_acceleration_search(data, method, ascan, cspan, X, Y):
    me = np.zeros((ele, cle))
    for i, ep in tqdm(enumerate(ascan)): # for acceleration
        if method.__name__ is 'vareig':
            datac = data * exp(-2j*pi*(ep*fa*X*Y*Y + ep*frs*Y*Y))
        elif method.__name__ is 'minimum_entropy':
            ep = ep
            datac = data * exp(-2j*pi*(ep*fa*X*Y*Y + ep*frs*Y*Y))
        else:
            print("Please Confirm your Method!")
            raise ValueError
        me[i, :] = method(cspan, datac, X, Y)[0]
    return me


method = vareig
me1 = angle_acceleration_search(data, method, ascan, cspan, X, Y)
plt.figure(figsize=[12, 8])
plt.imshow((-normalize(me1)), aspect='auto', cmap='gray', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.contour(-np.flipud((normalize(me1))), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.colorbar()
plt.scatter(vr, a, marker='x', s=60, label="Target1", c='r')
if save_fig:
    plt.savefig(method.__name__ + '{}.png'.format(SNR), dpi=300)

method = minimum_entropy
me2 = angle_acceleration_search(data, method, ascan, cspan, X, Y)
plt.figure(figsize=[12, 8])
plt.imshow(-normalize(me2), aspect='auto', cmap='gray', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.colorbar()
plt.scatter(vr, a, marker='x', s=60, label="Target1", c='r')
if save_fig:
    plt.savefig(method.__name__ + '{}.png'.format(SNR), dpi=300)


plt.figure(figsize=[12, 8])
plt.imshow(-(normalize(me2))*(normalize(me1)), aspect='auto', cmap='gray', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.colorbar()
plt.scatter(vr, a, marker='x', s=60, label="Target1", c='r')







# def keystone():
#      pass
#
#
ec, _ = minimum_entropy(cspan, data, X, Y)
# es, com = varsvd(cspan, data, X, Y)
ee, _ = vareig(cspan, data, X, Y)

plt.figure(figsize=[12, 8])
plt.plot(vspan, normalize(ec), label="Entropy of FFT", lw=2, color='g')
plt.plot(vspan, normalize(ee), label="Entropy of EIG", lw=2, color='r')
plt.vlines(np.array(vr), ymin = -0.5, ymax=1.5, linestyle='--', color='b')
plt.legend(loc='lower left')
plt.xlabel("Velocity (m/s)")
if save_fig:
    plt.savefig("MEvsVSVD.png", dpi=300)


# %% CLEAN technique

# method = vareig  #
# erot = 2
# new_data = data.copy()
# _, new_com = method(cspan, data, X, Y)
#
# indx1, indy1, new_data = CLEAN(data = new_data * exp(2j*pi*new_com*X*Y), zoom=2, erot=2.5).clean()
# new_data = new_data * exp(-2j*pi*new_com*X*Y)
# plt.figure()
# plt.imshow(20 * log10(abs(fft2(new_data * exp(2j * pi * new_com * X * Y)))), aspect='auto', cmap='jet')
# plt.colorbar()
#
#
# if number_target >=2:
#     _, new_com = method(cspan, new_data, X, Y)
#     indx2, indy2, new_data = CLEAN(data = new_data * exp(2j*pi*new_com*X*Y), zoom=2, erot=erot).clean()
#     new_data = new_data * exp(-2j*pi*new_com*X*Y)
#
#     plt.figure()
#     plt.imshow(20*log10(abs(fft2(new_data * exp(2j*pi*new_com*X*Y)))), aspect='auto', cmap='jet')
#     plt.colorbar()
#
# if number_target >=3:
#     _, new_com = method(cspan, new_data, X, Y)
#     indx3, indy3, new_data = CLEAN(data = new_data * exp(2j*pi*new_com*X*Y), zoom=2, erot=erot).clean()
#
#     # compare SE
# plt.figure(figsize=[12, 8])
# plt.scatter(indx1, indy1, marker='o', s=60, label="Target1")
#
# if number_target >= 2:
#     plt.scatter(indx2, indy2, marker='o', s=60, label="Target2")
#
# if number_target >= 3:
#     plt.scatter(indx3, indy3, marker='o', s=60, label="Target3")
#
# plt.scatter(-round_range1, -round_velocity1, marker='x', c='k', label="True Pos")
# plt.scatter(-round_range3, -round_velocity3, marker='x', c='k')
# plt.scatter(-round_range2, -round_velocity2, marker='x', c='k')
#
# plt.legend(loc="upper center")
# plt.xlabel("X")
# plt.ylabel("Y")
# if save_fig:
#     plt.savefig("ME.png", dpi=300)



plt.show()
