#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:58:07 2019

@author: shengzhixu
"""
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pylab import pi, exp, fft, fft2, log10, fftshift, sin, cos, deg2rad
from scipy.constants import speed_of_light as c
from scipy.io import loadmat
from skimage.transform import hough_line
from tqdm import tqdm

from multi_isar.CLEAN import CLEAN
from multi_isar.utils import cart2pol, pol2cart, awgn, normalize, entropy, renyi, tsallis, image_constrast

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
matplotlib.use('TkAgg')

# %% settings of simulation
save_fig = False
CMAP = plt.cm.hot


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

def rotate_target(Xc, Yc, theta=0):
    theta0 = deg2rad(theta)
    rho, phi = cart2pol(Xc, Yc)
    Xnew, Ynew = pol2cart(rho, phi + theta0)

    return Xnew, Ynew

Xc, Yc = rotate_target(Xc, Yc, 40)

plt.figure()
plt.scatter(Xc, Yc)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
if save_fig:
    plt.savefig("Target.png", dpi=300)

number_scatters = Xc.size

# c = - \mu * 2 * v * T / c / fs

# %% Radar parameters
B = 4e9     # bandwidth
resolution = c/2/B
fc = 78e9   # carrier frequency
T = 1e-3
Td = 0.8e-3

max_unambiguous_velocity = c/4/T/fc

Ts = Td/m
fs = 1/Ts
mu = B/Td
vm = c/4/T/fc


# %% Targets parameters
number_target = 3
R = [10.000, 10.677, 11.823]        # inital range
v = np.array([14, 12.5, 9]) * 1        # velocity ()
a = np.array([19, 17, 21]) * 0.80         # acceleration
# R = [10.000, 10.677, 11.823]   * 10     # inital range
# v = np.array([14, -12.5, 9]) * 10        # velocity ()
# a = np.array([19, 17, 21]) * 1.80         # acceleration
ele = 101                 # number of the searching grids for acceleration
cle = 101                                   # number of the searching grids for velocity

theta = [30, 30, 30]    # angle (should be similar)
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


low_alpha = 1       # variation of the amplitude (real)     # X:fast time
for i in range(number_scatters):
    data1 = data1 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[0]+Yc[i]) * X +     # range
                                 fd*(vr[0]+w[0]*Xc[i]) * Y +    # Doppler
                                 ar[0] * fa * X * Y * Y +   #
                                 ar[0] * frs * Y * Y +
                                 c1 * X * Y))


Xc, Yc = rotate_target(Xc, Yc, 0)
for i in range(number_scatters):
    data2 = data2 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[1]+Yc[i]) * X +
                                 fd*(vr[1]+w[1]*Xc[i]) * Y +
                                 ar[1] * fa * X * Y * Y +
                                 ar[1] * frs * Y * Y +
                                 c2 * X * Y))

Xc, Yc = rotate_target(Xc, Yc, 0)
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
dataf = fftshift(fft((data)* exp(2j*pi*c1*X*Y), axis=-1, n=1*m), axes=-1)
plt.figure(figsize=[12, 8])
plt.imshow(20*log10(abs(dataf)), aspect='auto', cmap=CMAP)
plt.clim(vmin=22, vmax=62)
plt.colorbar()
if save_fig:
    plt.savefig("1DFFT.png", dpi=300)

# # %%Hough Line Transform
# h, theta, d = hough_line(abs(dataf))
# plt.figure()
# plt.imshow(20*log10(abs(h + 1)), aspect='auto', cmap='gray')

# %%
dataf = fftshift(fft2((data)* exp(2j*pi* ( c1*X*Y + ar[0]*fa*X*Y*Y + ar[0]*frs*Y*Y)), [4*n, 4*m]))
plt.figure(figsize=[12, 8])
plt.imshow(20*log10(abs(dataf)), aspect='auto', cmap=CMAP)
plt.clim(vmin=50, vmax=90)
plt.colorbar()
if save_fig:
    plt.savefig("2DFFT.png", dpi=300)


#%% TEST STFT

# nperseg = 128
# noverlap = 127
# NFFT = 512
# f, t, Sxx = sp.signal.stft(data[0, :], fs=fs,
#                         nperseg=nperseg, noverlap=noverlap, return_onesided=True,
#                         nfft=NFFT)
# sp.signal.stft
#
# plt.imshow(abs(Sxx), aspect='auto', cmap=CMAP)

# %% Using ME or VSVD to separate targets and estimate the couplings

vspan = np.linspace(0, 15, cle)
cspan = vspan * fdr


def entropy_Fourier(cspan, data, X, Y, algorithm, alpha=1):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        isar = abs(fft2(datac, [1*n, 1*m]))
        ec[i] = algorithm(isar, alpha)
    print("Time for ME: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(ec) == ec).flatten()[0]
    cvalue = cspan[indc]

    return ec, cvalue



def varsvd(cspan, data, X, Y, alpha=1):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        es[i] = entropy(np.linalg.svd(datac, compute_uv=False))
    print("Time for SVD: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue


def entropy_eigen(cspan, data, X, Y, algorithm, alpha=1):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        es[i] = algorithm((np.linalg.eigvalsh(datac.T.conj().dot(datac))), alpha)
    print("Time for EIG: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue


ascan = np.linspace(-1.5*np.max(abs(a)), 1.5*np.max(abs(a)), ele)
# fascan = ascan * fa


def angle_acceleration_search(data, method, ascan, cspan, X, Y, algorithm, alpha=1):
    me = np.zeros((ele, cle))
    for i, ep in tqdm(enumerate(ascan)): # for acceleration
        if method.__name__ is 'entropy_eigen':
            # ep = 0
            datac = data * exp(-2j*pi*(ep*fa*X*Y*Y + ep*frs*Y*Y))
        elif method.__name__ is 'entropy_Fourier':
            ep = ep
            datac = data * exp(-2j*pi*(ep*fa*X*Y*Y + ep*frs*Y*Y))
        else:
            print("Please Confirm your Method!")
            raise ValueError
        me[i, :] = method(cspan, datac, X, Y, algorithm, alpha)[0]

    return me

#%%
method1 = entropy_eigen
algorithm = image_constrast
alpha = 1
me0 = angle_acceleration_search(data, method1, ascan, cspan, X, Y, algorithm=algorithm, alpha=alpha)

method1 = entropy_eigen
algorithm = renyi
alpha = 1
me1 = angle_acceleration_search(data, method1, ascan, cspan, X, Y, algorithm=algorithm, alpha=alpha)

method2 = entropy_Fourier
algorithm = renyi
alpha = 1
me2 = angle_acceleration_search(data, method2, ascan, cspan, X, Y, algorithm=algorithm, alpha=alpha)

method2 = entropy_Fourier
algorithm = image_constrast
alpha = 1
me3 = angle_acceleration_search(data, method2, ascan, cspan, X, Y, algorithm=algorithm, alpha=alpha)

# %% plt
plt.figure(figsize=[12, 8])
plt.imshow((1-normalize(me0)), aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud((normalize(me1))), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('Eig with IC')
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')
if save_fig:
    plt.savefig(method1.__name__ + '{}.png'.format(SNR), dpi=300)

plt.figure(figsize=[12, 8])
plt.imshow((normalize(me1)), aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud((normalize(me1))), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('Eig with Entropy')
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')
if save_fig:
    plt.savefig(method1.__name__ + '{}.png'.format(SNR), dpi=300)

plt.figure(figsize=[12, 8])
plt.imshow(normalize(me2), aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('Fourier with Entropy')
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')
if save_fig:
    plt.savefig(method2.__name__ + '{}.png'.format(SNR), dpi=300)

plt.figure(figsize=[12, 8])
plt.imshow(1-normalize(me3), aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('Fourier with IC')
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')
if save_fig:
    plt.savefig(method2.__name__ + '{}.png'.format(SNR), dpi=300)

#%% COMBINE
plt.figure(figsize=[12, 8])
plt.imshow((-( 0 + normalize(me3))*(1 - normalize(me1))),
           aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('EigEntropy + FourierIC')
plt.clim(-1, 0)
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')

plt.figure(figsize=[12, 8])
plt.imshow((-(1 - normalize(me2))*(1 - normalize(me1))),
           aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('EigEntropy + FourierEntropy')
plt.clim(-1, 0)
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')

plt.figure(figsize=[12, 8])
plt.imshow((-(1 - normalize(me2))*(0 + normalize(me0))),
           aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
# plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
plt.title('EigIC + FourierEntropy')
# plt.clim(-1, 0)
plt.colorbar()
plt.scatter(vr, ar, marker='o', s=100, facecolors='none', label="Target1", edgecolors='b')

def max_pos(map):
    '''
    Find the index of the maximum value of a 2D array
    :param map: 2D array
    :return: 2 index
    '''

    return np.unravel_index(map.argmax(), map.shape)



#%% for 1D velocity plot

# ec, _ = entropy_Fourier(cspan, data, X, Y, renyi)
# ec_renyi, _ = entropy_Fourier(cspan, data, X, Y, renyi)
#
# # es, com = varsvd(cspan, data, X, Y)
# ee, _ = entropy_eigen(cspan, data, X, Y, renyi)
# ee_renyi, _ = entropy_eigen(cspan, data, X, Y, renyi)
#
# plt.figure(figsize=[12, 8])
# plt.plot(vspan, normalize(ec), label="Entropy of FFT Shannon", lw=2, color='g')
# plt.plot(vspan, normalize(ec_renyi), label="Entropy of FFT Renyi", lw=2, color='r')
# plt.plot(vspan, normalize(ee), label="Entropy of EIG Shannon", lw=2, color='b')
# plt.plot(vspan, normalize(ee_renyi), label="Entropy of EIG Renyi", lw=2, color='c')
# plt.vlines(np.array(vr), ymin = -0.5, ymax=1.5, linestyle='--', color='b')
# plt.legend(loc='lower left')
# plt.xlabel("Velocity (m/s)")
# if save_fig:
#     plt.savefig("MEvsVSVD.png", dpi=300)


# %% CLEAN technique
#
# method = entropy_eigen  #
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
