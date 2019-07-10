#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:58:07 2019

@author: shengzhixu
"""
import time
import sys

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


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

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
# matplotlib.use('TkAgg')

# %% settings of simulation
save_fig = False
# CMAP = plt.cm.jet


# %% setting basic parameters
n = 256  # slow time
m = 256  # fast time
SNR = -5
[X, Y] = np.meshgrid(np.arange(n), np.arange(m))  # fast time, slow time
# model_zoom = 3.5

# %%
# fight = loadmat('/Users/shengzhixu/Documents/Python/multi_isar/Fighter2.mat')
# Xc = fight['Xc'].flatten()
# Yc = fight['Yc'].flatten()
car = np.load('/Users/shengzhixu/Documents/Python/multi_isar/car.npz')
Xc = car['arr_0']
Yc = car['arr_1']
Xc = (Xc - np.max(Xc)//2) / np.max(Xc) * 5
Yc = (Yc - np.max(Yc)//2) / np.max(Yc) * 2.5

Xc = Xc[::2]
Yc = Yc[::2]
plt.figure()
plt.scatter(Xc, Yc)
plt.xlabel("Xc")
plt.ylabel("Yc")
# Xc is the length ≈ 4.23meter (Xc.max()-Xc.min()), Yc is the width ≈ 2.10meter (Yc.max() - Yc.min())

#%%


def rotate_target(Xc, Yc, rotation=0):
    theta0 = deg2rad(rotation)
    rho, phi = cart2pol(Xc, Yc)
    Xnew, Ynew = pol2cart(rho, phi + theta0)

    return Xnew, Ynew

# theta1 = 30

#
# plt.figure()
# plt.scatter(Xcr, Ycr)
# plt.xlabel("X")
# plt.ylabel("Y")
# if save_fig:
#     plt.savefig("Target.png", dpi=300)

number_scatters = Xc.size

# c = - \mu * 2 * v * T / c / fs

# %% Radar parameters
B = 4e9     # bandwidth
resolution = c/2/B
fc = 78e9   # carrier frequency
T = 2e-4
Td = 0.8*T

max_unambiguous_velocity = c/4/T/fc

Ts = Td/m
fs = 1/Ts
mu = B/Td
vm = c/4/T/fc

'''
a = g = 10 m/s
0.5*a*(T*M)**2 = 0.5 * 10 * (1e-4 * 256)**2 = 0.0032768
dr = c/2/B = 3e8/2/4e9 = 0.0375

Acceleration rate can be ignored!!!!!!!!!
'''

# %% Targets parameters
range_domain = resolution * m
ambiguity = 1.5
delta_velocity = ambiguity * max_unambiguous_velocity


# number_target = 3
R = np.array([10.000, 10 - 0.15*range_domain, 10 + 0.2*range_domain, 10 + 0.7*range_domain]) - 0.5       # inital range
# v = np.array([14, 14, 14, 14]) * 0.98 + np.array([0, 1.05*delta_velocity, 1.4*delta_velocity, 0])      # velocity ()
# a = np.array([19, 19, -12, -12]) * 0.80         # acceleration
R = [14.619, 14.19, 10, 10]

# ## same velocity
number_target = 2
random = np.random.randn(4)
v = (np.array([70, 80, 14, 14]) + 70) / 3.6 + random * 0.5 # velocity ()
a = np.array([0, 0, 10, 10]) + np.random.rand(4) * 10

# v = np.array([43.22463802, 43.45906502, 22.6877857 , 23.84281342])
# a = np.array([-3.95944986,  4.40142665, 10.00087471, 10.02906805])


# ## same acceleration
# number_target = 2
# v = np.array([14, 14, 14, 14]) * 0.98 + np.array([0 , 1.05*delta_velocity, 1.4*delta_velocity, 0])     # velocity ()
# a = np.array([19, 19, 19, 19]) * 0.80         # acceleration

# one target entropy spread map
# number_target = 1
# R = [10.000, 10 - 0.2*range_domain, 10 + 0.2*range_domain, 10 + 0.7*range_domain]        # inital range
# v = np.array([16, 14, 14, 14]) * 0.96 + np.array([0, 1.05*delta_velocity, 1.4*delta_velocity, 0])      # velocity ()
# a = np.array([19, 19, -12, -12]) * 0.80         # acceleration


# %%

number_target = 2
random = np.random.randn(4)
v = (np.array([72, 80, 14, 14]) + 70) / 3.6 + random * 2 # velocity ()
a = np.array([0, 0, 10, 10]) + np.random.rand(4) * 2

# v = np.array([42.17461925, 40.6512066 , 20.23453959, 22.938097  ])
# a = np.array([ 0.97821432,  0.11965543, 10.73977276, 11.562339  ])

theta = [20, 35, 30, 30]    # angle (should be similar)
w = [0, 0, 0, 0]           # rotational velocity
vr = v*cos(deg2rad(theta))  # radial velocity
ar = a*cos(deg2rad(theta))
print(vr)
vt = v*sin(deg2rad(theta))  # translational velocity
# ascan = np.array([ar[0]-1, ar[0], ar[0]+1])

w = w + vt/R            # rotational velocity + translational_velocity / range


ele = 81                                    # number of the searching grids for acceleration
cle = 201                                   # number of the searching grids for velocity
vspan = np.linspace(np.min(vr)-2, np.max(vr)+2, cle)
ascan = np.linspace(-10, 10, ele)

# %% Generating data
data1 = np.zeros((m, n), dtype=complex)
data2 = np.zeros_like(data1, dtype=complex)
data3 = np.zeros_like(data1, dtype=complex)
data4 = np.zeros_like(data1, dtype=complex)

fr = mu*2/c/fs              # k    * y
fd = fc*T*2/c               # m    * x
frs = fc/c*T**2             # m^2  * a
fdr = mu*2*T/c/fs           # mk   * v
fa = mu/c*T**2/fs * 1       # km^2 * a


# Xij = 2*(R + Xcr)/c
# Yij = 2*(vr + Ycr*w)/c
# Zij = ar/c
# fr = mu*2*Xij/fs              # k    * y
# fd = fc*T*Yij/c               # m    * x
# frs = fc*T**2*Zij             # m^2  * a
# fdr = mu*2*T*Yij/fs           # mk   * v
# fa = mu*Zij*T**2/fs      # km^2 * a

Cr = fdr * vr


def fold(value):
    reminder = value - np.floor(value + 0.5)
    fold_num = np.floor(value)
    return reminder, fold_num

def varing_amplitude(low_alpha):
    real = low_alpha

Xcr, Ycr = rotate_target(Xc, Yc, theta[0])
low_alpha = 0.5      # variation of the amplitude (real)     # X:fast time
for i in range(number_scatters):
    data1 = data1 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[0]+Xcr[i]) * X +     # range
                                 fd*(vr[0] + w[0]*Ycr[i]) * Y +    # Doppler
                                 ar[0] * fa * X * Y * Y +   #
                                 ar[0] * frs * Y * Y +
                                 Cr[0] * X * Y))
round_range1, _ = fold((R[0] + Ycr)*fr)
round_velocity1, fold1 = fold((vr[0] + w[0]*Xcr)*fd)


Xcr1, Ycr1 = rotate_target(Xc, Yc, theta[1])
for i in range(number_scatters):
    data2 = data2 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[1]+Xcr1[i]) * X +
                                 fd*(vr[1] + w[1]*Ycr1[i]) * Y +
                                 ar[1] * fa * X * Y * Y +
                                 ar[1] * frs * Y * Y +
                                 Cr[1] * X * Y))
round_range2, _= fold((R[1] + Ycr1)*fr)
round_velocity2, fold2 = fold((vr[1] + w[1]*Xcr1)*fd)

Xcr, Ycr = rotate_target(Xcr, Ycr, 0)
for i in range(number_scatters):
    data3 = data3 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[2]+Xcr[i]) * X +
                                 fd*(vr[2]+w[2]*Ycr[i]) * Y +
                                 ar[2] * fa * X * Y * Y +
                                 ar[2] * frs * Y * Y +
                                 Cr[2] * X * Y))
round_range3, _ = fold((R[2] + Ycr)*fr)
round_velocity3, fold3 = fold((vr[2] + w[2]*Xcr)*fd)


Xcr, Ycr = rotate_target(Xcr, Ycr, )
for i in range(number_scatters):
    data4 = data4 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j*pi*( fr*(R[3]+Xcr[i]) * X +
                                 fd*(vr[3]+w[3]*Ycr[i]) * Y +
                                 ar[3] * fa * X * Y * Y +
                                 ar[3] * frs * Y * Y +
                                 Cr[3] * X * Y))
round_range4, _ = fold((R[3] + Ycr)*fr)
round_velocity4, fold4 = fold((vr[3] + w[3]*Xcr)*fd)


if number_target == 4:
    data = data1 + + data2 + data3 + data4
elif number_target == 3:
    data = data1 + data2 + data3
elif number_target == 2:
    data = data1 + data2
else:
    data = data1

data = awgn(data, SNR)   # X (range), Y (Doppler)

#%%

data1f = fftshift(fft((data)* exp(2j*pi*Cr[1]*X*Y), axis=-1, n=4*m), axes=-1)
plt.figure(figsize=[8, 5])
plt.imshow(20*log10(abs(data1f)), aspect='auto', cmap='jet', extent=[0, range_domain, 0, m])
plt.clim(vmin=22, vmax=62)
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.xlabel('Range (m)')
plt.ylabel('Slow Time Bins')
if save_fig:
    plt.savefig("1DFFT_{}.png".format(number_target), dpi=300)

# Observed Scene



# %%Hough Line Transform
# h, theta, d = hough_line(abs(data1f))
# plt.figure()
# # plt.imshow(20*log10(abs(h + 1)), aspect='auto', cmap='gray')
# plt.imshow(h, aspect='auto', cmap='gray')
# plt.colorbar()

# %% centroid Doppler compensation
# com_v = -0.5*max_unambiguous_velocity
# data = data * exp(2j * pi * com_v * fd * Y)

# %% image the scene

def scene(Xc, Yc, theta, R):
    Xt, Yt = rotate_target(Xc, Yc, theta)
    new_Xt = Xt + R
    Xtb, Ytb = rotate_target(new_Xt, Yt, -theta)

    return Xtb, Ytb


plt.figure(figsize=[8, 8])
x1, y1 = scene(Xc, Yc, theta[0], R[0])
x2, y2 = scene(Xc, Yc, theta[1], R[1])
plt.scatter(y1, x1, s=2, label='car 1')
plt.scatter(y2, x2, s=2, label='car 2')
plt.scatter([0], [0], s=300, marker="v", label='radar')
plt.xlim((-10, 10))
plt.ylim((0, 20))
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(axis='x', ls=":")
# plt.legend()

# %%
I = 0
dataf = (fft2((data)* exp(1 *2j*pi* ( Cr[I]*X*Y + ar[I]*fa*X*Y*Y + ar[I]*frs*Y*Y)), [4*n, 4*m]))
plt.figure(figsize=[8, 5])
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[0, range_domain, -max_unambiguous_velocity, max_unambiguous_velocity],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-20, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.xlabel('Range (m)')
plt.ylabel('Doppler (m/s)')
if save_fig:
    plt.savefig("2DFFT_{}.png".format(number_target), dpi=300)


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


cspan = vspan * fdr


def Fourier(cspan, data, X, Y, algorithm, alpha=1):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        isar = abs(fft2(datac, [1*n, 1*m]))
        ec[i] = algorithm(isar, alpha)
    print("Time for FFT: {:.3f} seconds".format(time.time()-tic))

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


def eigen(cspan, data, X, Y, algorithm, alpha=1):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        eigvals = (np.linalg.eigvalsh(datac.T.conj().dot(datac)))
        es[i] = algorithm(eigvals, alpha)
    print("Time for EIG: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue



# fascan = ascan * fa


def angle_acceleration_search(data, method, ascan, cspan, X, Y, algorithm, alpha=1):
    me = np.zeros((ele, cle))
    for i, ep in tqdm(enumerate(ascan)): # for acceleration
        if method.__name__ is 'eigen':
            # ep = 0
            datac = data * exp(2j*pi*(ep*fa*X*Y*Y + ep*frs*Y*Y))
        elif method.__name__ is 'Fourier':
            ep = ep
            datac = data * exp(2j*pi*(ep*fa*X*Y*Y + ep*frs*Y*Y))
        else:
            print("Please Confirm your Method!")
            raise ValueError
        me[i, :] = method(cspan, datac, X, Y, algorithm, alpha)[0]

    return me

#%%
# m0: EigIC
# m1: EigEP
# m2: FourierEP
# m3: FourierIC

me0, me1, me2, me3 = None, None, None, None
method1, method2, method3, method0 = None, None, None, None
algorithm1, algorithm2, algorithm3, algorithm0 = None, None, None, None

# method0 = eigen
# algorithm0 = image_constrast

method1 = eigen
algorithm1 = image_constrast

method2 = Fourier
algorithm2 = algorithm1

# method3 = Fourier
# algorithm3 = image_constrast
alpha = 1



# me0 = angle_acceleration_search(data, method0, ascan, cspan, X, Y, algorithm=algorithm0, alpha=alpha)
me1 = angle_acceleration_search(data, method1, ascan, cspan, X, Y, algorithm=algorithm1, alpha=alpha)
me2 = angle_acceleration_search(data, method2, ascan, cspan, X, Y, algorithm=algorithm2, alpha=alpha)
# me3 = angle_acceleration_search(data, method3, ascan, cspan, X, Y, algorithm=algorithm3, alpha=alpha)

# %% plt
if algorithm1 == renyi:
    CMAP = 'jet_r'
else:
    CMAP = 'jet'

if me0 is not None:
    plt.figure(figsize=[8, 5])
    plt.imshow((1-normalize(me0)), aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud((normalize(me1))), aspect='auto', cmap='jet',
    #             extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='b')
    if save_fig:
        plt.savefig(method0.__name__ + algorithm0.__name__ + '{}_{}.png'.format(SNR, number_target), dpi=300)
    plt.title('Eig with IC')

if me1 is not None:
    plt.figure(figsize=[8, 5])
    plt.imshow(np.flipud(normalize(me1[:, :])), aspect='auto', cmap=CMAP,
               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud((normalize(me1))), aspect='auto', cmap='jet',
    #             extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='w',
                linewidths=3)
    if save_fig:
        plt.savefig(method1.__name__ + algorithm1.__name__ + '{}_{}.png'.format(SNR, number_target), dpi=300)
    plt.title('Eig with Entropy')

if me2 is not None:
    plt.figure(figsize=[8, 5])
    plt.imshow(np.flipud(normalize(me2)), aspect='auto', cmap=CMAP,
               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='w',
                linewidths=3)
    if save_fig:
        plt.savefig(method2.__name__ + algorithm2.__name__ + '{}_{}.png'.format(SNR, number_target), dpi=300)
    plt.title('Fourier with Entropy')

if me3 is not None:
    plt.figure(figsize=[8, 5])
    plt.imshow(1-normalize(me3), aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet',
    #               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='b')
    if save_fig:
        plt.savefig(method3.__name__ + algorithm3.__name__ + '{}_{}.png'.format(SNR, number_target), dpi=300)
    plt.title('Fourier with IC')



#%% COMBINE
# m0: EigIC
# m1: EigEP
# m2: FourierEP
# m3: FourierIC

if (me3 is not None) and (me1 is not None):
    plt.figure(figsize=[8, 5])
    plt.imshow(-(( 0 + normalize(me3))*(1 - normalize(me1))),
               aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet',
    #              extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    # plt.clim(-1, 0)
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='b',
                linewidths=1)
    if save_fig:
        plt.savefig('EigEntropy_FourierIC_{}.png'.format(number_target), dpi=300)
    plt.title('EigEP + FourierIC')


if (me2 is not None) and (me1 is not None):
    if algorithm2 == renyi:
        me_com = -(1 - normalize(me2)) * (1 - normalize(me1))
    else:
        me_com = normalize(me1) * normalize(me2)
    plt.figure(figsize=[8, 5])
    plt.imshow(np.flipud(me_com),
               aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet',
    #               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    # plt.clim(0, 1)
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='w',
                linewidths=3)
    if save_fig:
        plt.savefig('EigEP_FourierEP_{}.png'.format(number_target), dpi=300)
    plt.title('EigEntropy + FourierEntropy')


if (me2 is not None) and (me0 is not None):
    plt.figure(figsize=[8, 5])
    plt.imshow(-((1 - normalize(me2))*(0 + normalize(me0))),
               aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet',
    #               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    # plt.clim(-1, 0)
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='b')
    if save_fig:
        plt.savefig('EigIC_FourierEP_{}.png'.format(number_target), dpi=300)
    plt.title('EigIC + FourierEntropy')

if (me3 is not None) and (me0 is not None):
    plt.figure(figsize=[8, 5])
    plt.imshow(-((0 + normalize(me3))*(0 + normalize(me0))),
               aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet',
    #               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    # plt.clim(-1, 0)
    plt.colorbar()
    plt.scatter(vr[0:number_target],
                ar[0:number_target],
                marker='o',
                s=150,
                facecolors='none',
                label="Target1",
                edgecolors='b')
    if save_fig:
        plt.savefig('EigIC_FourierEP_{}.png'.format(number_target), dpi=300)
    plt.title('EigIC + FourierIC')



def max_pos(map):
    '''
    Find the index of the maximum value of a 2D array
    :param map: 2D array
    :return: 2 index
    '''

    return np.unravel_index(map.argmax(), map.shape)

# %%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# V, A = np.meshgrid(vspan, ascan)
# surf = ax.plot_surface(V, A, -((1 - normalize(me2))*(1 - normalize(me1))), cmap='jet',
#                        linewidth=0, antialiased=True)
#
# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)




#%% for 1D velocity plot

# ec, _ = Fourier(cspan, data, X, Y, renyi)
# ec_renyi, _ = Fourier(cspan, data, X, Y, renyi)
#
# # es, com = varsvd(cspan, data, X, Y)
# ee, _ = eigen(cspan, data, X, Y, renyi)
# ee_renyi, _ = eigen(cspan, data, X, Y, renyi)
#
# plt.figure(figsize=[8, 5])
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

'''
a more simple way to image two targets is using the Thresholding
'''
if algorithm2 == renyi:
    indx_a, indx_v = max_pos(-me1)
else:
    indx_a, indx_v = max_pos(me1)
a_est1 = ascan[indx_a]
v_est1 = vspan[indx_v]
print("velocity1: {}    acceleration1: {}".format(v_est1, a_est1))


if algorithm2 == renyi:
    indx_a, indx_v = max_pos(-me2)
else:
    indx_a, indx_v = max_pos(me2)
a_est2 = ascan[indx_a]
v_est2 = vspan[indx_v]
print("velocity2: {}    acceleration2: {}".format(v_est2, a_est2))


if algorithm2 == renyi:
    me_com = -(1 - normalize(me2)) * (1 - normalize(me1))
    indx_a, indx_v = max_pos(-me_com)
else:
    me_com = normalize(me1) * normalize(me2)
    indx_a, indx_v = max_pos(me_com)

a_est3 = ascan[indx_a]
v_est3 = vspan[indx_v]
print("velocity3: {:.4}    acceleration3: {:.4}".format(v_est3, a_est3))


v_est = v_est3
a_est = a_est3

# %% Thresholding

dataf = (fft2((data)* exp(1 *2j*pi* ( v_est*fdr*X*Y + a_est*fa*X*Y*Y + a_est*frs*Y*Y)), [2*n, 2*m]))
fig = plt.figure(figsize=[16,5])
fig.add_subplot(1, 2, 1)
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[0, range_domain, -max_unambiguous_velocity, max_unambiguous_velocity],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-20, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.xlabel('Range (m)')
plt.ylabel('Doppler (m/s)')
if save_fig:
    plt.savefig("2DFFT_{}.png".format(number_target), dpi=300)


from skimage import morphology
dilation = morphology.binary_dilation
max_spec = np.max(data_fft2_db)
threshold = 5 #db
# fig.add_subplot(1, 3, 2)
car1 = (data_fft2_db > (max_spec-threshold))
# plt.imshow(car1,
#            interpolation='none',
#            cmap='gray_r')
fig.add_subplot(1, 2, 2)
car_dilation = dilation(dilation(car1))
plt.imshow(np.fliplr(car_dilation.T),
           aspect='auto',
           extent=[0, range_domain, -max_unambiguous_velocity, max_unambiguous_velocity],
           interpolation='none',
           cmap='gray_r')
plt.xlabel('Range (m)')
plt.ylabel('Doppler (m/s)')

# erosion = morphology.binary_erosion
# car2 = erosion(erosion(data_fft2_db < (max_spec-threshold))) * dataf
# plt.figure()
# plt.imshow(car2,
#            cmap='jet',
#            aspect='auto')
# plt.clim(vmin=np.max(data_fft2_db)-20, vmax=np.max(data_fft2_db))
# car2_rev = car2 * exp( -2j*pi* ( v_est*fdr*X*Y + a_est*fa*X*Y*Y + a_est*frs*Y*Y))
# data = car2_rev



# %% CLEAN Techinque
# new_data = data
# com_term = exp(2j*pi* ( v_est*fdr*X*Y + a_est*fa*X*Y*Y + a_est*frs*Y*Y))
# com_data = new_data * com_term
# indx1, indy1, new_data = CLEAN(data = com_data, zoom=2, erot=2).clean()
# plt.figure(figsize=[8, 5])
# plt.scatter(indx1, indy1, marker='x', s=60, label="Target1")
# new_data = new_data * com_term.conj()


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

#
# if number_target >= 2:
#     plt.scatter(indx2, indy2, marker='o', s=60, label="Target2")
#
# if number_target >= 3:
#     plt.scatter(indx3, indy3, marker='o', s=60, label="Target3")





# %% fold Doppler
r0 = (R[0] + Ycr)
v0 = (vr[0] + w[0]*Xcr)
r1 = (R[1] + Ycr)
v1 = (vr[1] + w[1]*Xcr)

fig = plt.figure(figsize=[7, 5])

plt.scatter(round_velocity1/fd, r0,  marker='x', c='k', label="True Pos1")
plt.scatter(round_velocity2/fd, r1, marker='x', c='g',  label="True Pos2")
# plt.scatter(-round_range3, -round_velocity3, marker='x', c='b',  label="True Pos3")
# plt.xlim((-0.5, 0.5))
# plt.ylim((-0.5, 0.5))
plt.legend(loc="upper left")
plt.xlabel("Folded Doppler velocity (m/s)")
plt.ylabel("Folded Range (m)")
if save_fig:
    plt.savefig("ME_{}.png".format(number_target), dpi=300)

# %% unfold
plt.figure(figsize=[7, 5])


plt.scatter(v0, r0, marker='x', c='k', label="Car 1")
plt.scatter(v1, r1, marker='x', c='g',  label="Car 2")
# plt.scatter(-round_range3, -round_velocity3, marker='x', c='b',  label="True Pos3")
# plt.xlim((-0.5, 0.5))
# plt.ylim((-0.5, 0.5))
plt.legend(loc="upper left")
plt.xlabel("Doppler velocity (m/s)")
plt.ylabel("Range (m)")
if save_fig:
    plt.savefig("ME_{}.png".format(number_target), dpi=300)


# sys.stdout.write('\a')
# sys.stdout.flush()
plt.show()
