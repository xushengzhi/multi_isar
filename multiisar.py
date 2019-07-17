#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:58:07 2019

@author: shengzhixu
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from pylab import pi, exp, fft, fft2, log10, fftshift, sin, cos, deg2rad
from scipy.constants import speed_of_light as c
from tqdm import tqdm

from multi_isar.Keystone import Keystone
from multi_isar.utils import cart2pol, pol2cart, awgn, normalize, entropy, renyi, tsallis, image_constrast
from multi_isar.detect_local_minima import detect_local_minima
from skimage.restoration import denoise_tv_chambolle


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

# %% settings of simulation
save_fig = False



# %% setting basic parameters
k = 200  # fast time
m = 156  # slow time
[K, M] = np.meshgrid(np.arange(k), np.arange(m))  # slow time * fast time (X is fasttime, Y is slowtime)
# 156*200     X(0->200), Y(0->156)

# %%

car = np.load('/Users/shengzhixu/Documents/Python/multi_isar/car.npz')
Xc = car['arr_0']
Yc = car['arr_1']
Xc = (Xc - np.max(Xc)//2) / np.max(Xc) * 5
Yc = (Yc - np.max(Yc)//2) / np.max(Yc) * 2.5

Xc = Xc[::2]
Yc = Yc[::2]

plt.figure()
plt.scatter(Yc, Xc)
plt.xlabel("Xc")
plt.ylabel("Yc")

#%%
def rotate_target(Xc, Yc, rotation=0):
    theta0 = deg2rad(rotation)
    rho, phi = cart2pol(Xc, Yc)
    Xnew, Ynew = pol2cart(rho, phi + theta0)

    return Xnew, Ynew

number_scatters = Xc.size

# c = - \mu * 2 * v * T / c / fs

# %% Radar parameters
B = 4e9     # bandwidth
resolution = c/2/B
fc = 78e9   # carrier frequency
T = 2.5e-4
Td = 0.8*T

max_unambiguous_velocity = c/4/T/fc

Ts = Td/k
fs = 1/Ts
mu = B/Td
vm = c/4/T/fc    #(-vm, vm)

'''
a = g = 10 m/s
0.5*a*(T*M)**2 = 0.5 * 10 * (3e-4 * 156)**2 = 0.010951
dr = c/2/B = 3e8/2/4e9 = 0.0375

Acceleration rate can be ignored!!!!!!!!!
'''

# %% Targets parameters
range_domain = resolution * k
R = [21.619, 20.69, 10, 10]
random = np.random.rand(4)  # add random velocity to the targets
v = (np.array([80, 65+3, 14, 14]) + 80) / 3.6 + random * 0.01 # velocity ()
a = np.array([-4, 4, 10, 10])*0 + np.random.rand(4) * 1

# !! v = (np.array([80, 65+8, 14, 14]) + 80) / 3.6 + random * 0.01 # velocity ()  # eigen >> image_contrast

theta = [20, 30, 30, 30]    # angle (should be similar)
w = [0, 0, 0, 0]           # rotational velocity
vr = v*cos(deg2rad(theta))  # radial velocity
ar = a*cos(deg2rad(theta))
print(vr)
vt = v*sin(deg2rad(theta))  # translational velocity
w = w + vt/R            # rotational velocity + translational_velocity / range

ele = 41                                    # number of the searching grids for acceleration
cle = 81                           # number of the searching grids for velocity

vspan = np.linspace(np.min(vr[0:2])-5, np.max(vr[0:2])+5, cle)
# vspan = np.arange(3, 9) * 2*max_unambiguous_velocity
keystone_usd = False
algorithm11 = renyi                   # eigen
algorithm22 = renyi                   # fourier
SNR = 5
low_alpha = 0.8      # variation of the amplitude (real)     # X:fast time
number_target = 2
ascan = np.linspace(-6, 6, ele)

# %% Generating data
data1 = np.zeros((m, k), dtype=complex)
data2 = np.zeros_like(data1, dtype=complex)
data3 = np.zeros_like(data1, dtype=complex)
data4 = np.zeros_like(data1, dtype=complex)

fr = mu*2/c/fs              # k    * y
fd = fc*T*2/c               # m    * x
frs = fc/c*T**2             # m^2  * a
fdr = mu*2*T/c/fs           # mk   * v
fa = mu/c*T**2/fs * 1       # km^2 * a

Cr = fdr * vr

def fold(value):
    reminder = value - np.floor(value + 0.5)
    fold_num = np.floor(value)
    return reminder, fold_num

with_spread = 1

Xcr, Ycr = rotate_target(Xc, Yc, theta[0])


def complex_amplitude(low_alpha):
    COF = np.sqrt(2)/2
    real = COF * (low_alpha + (1-low_alpha) * np.random.rand())
    imag = COF * (low_alpha + (1-low_alpha) * np.random.rand())
    return real+ 1j * imag


for i in range(number_scatters):
    data1 = data1 + complex_amplitude(low_alpha) * \
                    exp(-2j * pi * (fr * (R[0]+Xcr[i]) * K +  # range
                                    fd * (vr[0] + w[0]*Ycr[i]) * M +  # Doppler
                                    ar[0] * fa * K * M * M +  #
                                    ar[0] * frs * M * M +
                                    (vr[0] + with_spread*w[0]*Ycr[i]) * fdr * K * M))
round_range1, _ = fold((R[0] + Ycr)*fr)
round_velocity1, fold1 = fold((vr[0] + w[0]*Xcr)*fd)


Xcr1, Ycr1 = rotate_target(Xc, Yc, theta[1])
for i in range(number_scatters):
    data2 = data2 + complex_amplitude(low_alpha) * \
                    exp(-2j * pi * (fr * (R[1]+Xcr1[i]) * K +
                                    fd * (vr[1] + w[1]*Ycr1[i]) * M +
                                    ar[1] * fa * K * M * M +
                                    ar[1] * frs * M * M +
                                    (vr[1] + with_spread*w[1]*Ycr[i]) * fdr * K * M))
round_range2, _= fold((R[1] + Ycr1)*fr)
round_velocity2, fold2 = fold((vr[1] + w[1]*Xcr1)*fd)

Xcr, Ycr = rotate_target(Xcr, Ycr, 0)
for i in range(number_scatters):
    data3 = data3 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j * pi * (fr * (R[2]+Xcr[i]) * K +
                                    fd * (vr[2]+w[2]*Ycr[i]) * M +
                                    ar[2] * fa * K * M * M +
                                    ar[2] * frs * M * M +
                                    Cr[2] * K * M))
round_range3, _ = fold((R[2] + Ycr)*fr)
round_velocity3, fold3 = fold((vr[2] + w[2]*Xcr)*fd)


Xcr, Ycr = rotate_target(Xcr, Ycr, )
for i in range(number_scatters):
    data4 = data4 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j * pi * (fr * (R[3]+Xcr[i]) * K +
                                    fd * (vr[3]+w[3]*Ycr[i]) * M +
                                    ar[3] * fa * K * M * M +
                                    ar[3] * frs * M * M +
                                    Cr[3] * K * M))
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

data_nonoise = data
data = awgn(data, SNR)
# data \in C^{ m*k }

#%% 1DFFT

data1f = fftshift(fft(data*(exp(0*2j * pi * (Cr[0] * K * M))) , axis=-1, n=4*k), axes=-1)
plt.figure(figsize=[8, 5])
data1f_db = 20 * log10(abs(data1f))
plt.imshow(np.flipud(data1f_db).T,
           aspect='auto',
           cmap='jet',
           extent=[0, m, 1.5*range_domain, 2.5*range_domain])
plt.clim(vmin=np.max(data1f_db) - 40, vmax=np.max(data1f_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Slow Time Bins')
if save_fig:
    plt.savefig("1DFFT_{}.png".format(number_target), dpi=300)

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
plt.xlim((-20, 20))
plt.ylim((0, 40))
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(axis='x', ls=":")
if save_fig:
    plt.savefig("scenario.png", dpi=300)
# plt.legend()

# %% 2DFFT
I = 0
# vr[I] = 36
comdata = data*exp(2j*pi*(vr[I]*fdr*K*M +ar[I]*fa*K*M*M + ar[I]*frs*M*M + vr[I]*fd*M))
# comdata = Keystone(comdata, fd, fdr)
dataf = fftshift((fft2(comdata, [4*m, 4*k])))
plt.figure(figsize=[8, 5])
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[-vm, vm, 0, 200*resolution],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-40, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("2DFFT_{}.png".format(number_target), dpi=300)


# %% Using ME or VSVD to separate targets and estimate the couplings
cspan = vspan
def Fourier(cspan, data, K, M, algorithm, alpha=1):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, v in (enumerate(cspan)):
        datac = data * exp(2j * pi * (v * fdr * K * M + v * fd * M))
        if keystone_usd:
            datac = Keystone(datac, fd, fdr, verbose=False)
        isar = abs(fft2(datac, [1 * k, 1 * m]))
        ec[i] = algorithm(isar, alpha)
    print("Time for FFT: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(ec) == ec).flatten()[0]
    cvalue = cspan[indc]

    return ec, cvalue


def eigen(cspan, data, K, M, algorithm, alpha=1):
    es = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, v in (enumerate(cspan)):
        datac = data * exp(2j * pi * (v * fdr * K * M + v * fd * M))
        if keystone_usd:
            datac = Keystone(datac, fd, fdr, verbose=False)
        eigvals = (np.linalg.eigvalsh(datac.dot(datac.T.conj())))
        es[i] = algorithm(eigvals, alpha)
    print("Time for EIG: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue


def angle_acceleration_search(data, method, ascan, cspan, K, M, algorithm, alpha=1):
    me = np.zeros((ele, cle))
    for i, a in tqdm(enumerate(ascan)): # for acceleration
        if method.__name__ is 'eigen':
            # a = 0
            datac = data * exp(2j * pi * (a * fa * K * M * M + a * frs * M * M))
        elif method.__name__ is 'Fourier':
            a = a
            datac = data * exp(2j * pi * (a * fa * K * M * M + a * frs * M * M))
        else:
            print("Please Confirm your Method!")
            raise ValueError
        me[i, :] = method(cspan, datac, K, M, algorithm, alpha)[0]

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
algorithm1 = algorithm11

method2 = Fourier
algorithm2 = algorithm22

# method3 = Fourier
# algorithm3 = image_constrast
alpha = 1

# me0 = angle_acceleration_search(data, method0, ascan, cspan, X, Y, algorithm=algorithm0, alpha=alpha)
me1 = angle_acceleration_search(data, method1, ascan, cspan, K, M, algorithm=algorithm1, alpha=alpha)
me2 = angle_acceleration_search(data, method2, ascan, cspan, K, M, algorithm=algorithm2, alpha=alpha)
# me3 = angle_acceleration_search(data, method3, ascan, cspan, X, Y, algorithm=algorithm3, alpha=alpha)
if algorithm1 == renyi:
    me1 = - me1
if algorithm2 == renyi:
    me2 = - me2
# %% plt
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


# %% fold Doppler
r0 = (R[0] + Xcr)
v0 = (vr[0] + w[0]*Ycr)
r1 = (R[1] + Xcr1)
v1 = (vr[1] + w[1]*Ycr1)
range_fold = R[0] // range_domain
doppler_fold = vr[0] // (2*max_unambiguous_velocity) * 2

fig = plt.figure(figsize=[7, 5])
plt.scatter(v0 % (2*max_unambiguous_velocity)-max_unambiguous_velocity, r0,  marker='x', c='k', label="True Pos1")
plt.scatter(v1 % (2*max_unambiguous_velocity)-max_unambiguous_velocity, r1, marker='x', c='g',  label="True Pos2")
plt.legend(loc="upper left")
plt.xlabel("Folded Doppler velocity ($m/s$)")
plt.ylabel("Range (m)")
plt.ylim((1.5)*range_domain, (2.5)*range_domain)
plt.xlim(-max_unambiguous_velocity, 1*max_unambiguous_velocity)
if save_fig:
    plt.savefig("folded_velocity_{}.png".format(number_target), dpi=300)

# %% unfold
plt.figure(figsize=[7, 5])
plt.scatter(v0, r0, marker='x', c='k', label="Car 1")
plt.scatter(v1, r1, marker='x', c='g',  label="Car 2")
plt.legend(loc="upper left")
plt.xlabel("Doppler velocity ($m/s$)")
plt.ylabel("Range (m)")
plt.ylim(2.5*range_domain, 3.5*range_domain)
plt.xlim((doppler_fold-3)*max_unambiguous_velocity, (doppler_fold+2)*max_unambiguous_velocity)
plt.vlines(x = (doppler_fold-2)*max_unambiguous_velocity, ymin=0, ymax=100, linestyles=':')
plt.vlines(x = (doppler_fold)*max_unambiguous_velocity, ymin=0, ymax=100, linestyles=':')
plt.vlines(x = (doppler_fold+2)*max_unambiguous_velocity, ymin=0, ymax=100, linestyles=':')
if save_fig:
    plt.savefig("unfolded_velocity_{}.png".format(number_target), dpi=300)




#%% Estimator
CMAP = 'hot_r'
scatter_c = 'w'
MARKER = 'X'

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharex=True, sharey=True)
weight = 0.5
np.set_printoptions(precision=2)

me2 = normalize(me2)
me1 = normalize(me1)
me_com = normalize(me_com)


ax[0, 0].imshow(me1, aspect='auto', cmap=CMAP)
ax[0, 0].axis('off')
# ax[0, 0].set_title("Me1")

ax[1, 0].imshow(denoise_tv_chambolle(me1), aspect='auto', cmap=CMAP)
ax[1, 0].axis('off')
# ax[1, 0].set_title("Me1 denoising")

arr1 = detect_local_minima(-denoise_tv_chambolle(me1, weight=weight))
indy, indx = arr1
to_delete_list = []
number_target_est = indx.size
for i in range(number_target_est):
    if me1[indy[i], indx[i]] <= 0.4:
        to_delete_list.append(i)
indx1 = np.delete(indx, to_delete_list)
indy1 = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx1], ascan[indy1]))
ax[1, 0].scatter(indx1, indy1, s=20, c=scatter_c, marker=MARKER)


ax[0, 1].imshow(me2, aspect='auto', cmap=CMAP)
ax[0, 1].axis('off')
# ax[0, 1].set_title("Me2")

ax[1, 1].imshow(denoise_tv_chambolle(me2), aspect='auto', cmap=CMAP)
ax[1, 1].axis('off')
# ax[1, 1].set_title("Me2 denoising")

arr2 = detect_local_minima(-denoise_tv_chambolle(me2, weight=weight))
indy, indx = arr2
to_delete_list = []
number_target_est = indx.size

for i in range(number_target_est):
    if me2[indy[i], indx[i]] <= 0.4:
        to_delete_list.append(i)
indx2 = np.delete(indx, to_delete_list)
indy2 = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx2], ascan[indy2]))
ax[1, 1].scatter(indx2, indy2, s=20, c=scatter_c, marker=MARKER)




ax[0, 2].imshow(me_com, aspect='auto', cmap=CMAP)
ax[0, 2].axis('off')
# ax[0, 2].set_title("MeCOM")

ax[1, 2].imshow(denoise_tv_chambolle(me_com), aspect='auto', cmap=CMAP)
ax[1, 2].axis('off')
# ax[1, 2].set_title("MeCOM denoising")

arr3 = detect_local_minima(-denoise_tv_chambolle(me_com, weight=weight))
indy, indx = arr3
to_delete_list = []
number_target_est = indx.size

for i in range(number_target_est):
    if me_com[indy[i], indx[i]] <= 0.4:
        to_delete_list.append(i)
indx3 = np.delete(indx, to_delete_list)
indy3 = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx3], ascan[indy3]))
ax[1, 2].scatter(indx3, indy3, s=20, c=scatter_c, marker=MARKER)

plt.tight_layout()

if save_fig:
    plt.savefig("denoising_estimation.png", dpi=400)


#%% estimation of parameters

v_est = vspan[indx3]
a_est = ascan[indy3] * 1

v_fold = v_est // (2*max_unambiguous_velocity) * 2

#%% thresholding
zoom = 4
dataf = (data)* exp(2j*pi*(v_est[0]*fdr*K*M + a_est[0]*fa*K*M*M + a_est[0]*frs*M*M + v_est[0]*fd * M))
# dataf = data* exp(2j*pi*(v_fold[0]*vm*fdr*K*M + a_est[0]*fa*K*M*M + a_est[0]*frs*M*M))
dataf = Keystone(dataf, fd, fdr);
dataf = fftshift(fft2(dataf, [zoom*m, zoom*k]))



fig = plt.figure(figsize=[16,5])
fig.add_subplot(1, 2, 1)
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[v_fold[0] * max_unambiguous_velocity,
                   (v_fold[0] +2 ) * max_unambiguous_velocity,
                    1.5*range_domain,
                   2.5*range_domain],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-40, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("first_focusing.png".format(number_target), dpi=300)

from skimage import morphology
dilation = morphology.binary_dilation
max_spec = np.max(data_fft2_db)
threshold = 5 #db
car1 = (data_fft2_db > (max_spec-threshold))
fig.add_subplot(1, 2, 2)
car_dilation = dilation(dilation(dilation(car1)))
plt.imshow(np.fliplr(car_dilation.T),
           aspect='auto',
           extent=[v_fold[0] * max_unambiguous_velocity,
                   (v_fold[0] +2 ) * max_unambiguous_velocity,
                    1.5*range_domain,
                   2.5*range_domain],
           interpolation='none',
           cmap='gray_r')
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("first_thresholding.png".format(number_target), dpi=300)


dataf = (data)*exp(2j*pi*(v_est[1]*fdr*K*M+a_est[1]*fa*K*M*M+a_est[1]*frs*M*M+ v_est[1]*fd * M))
dataf = Keystone(dataf, fd, fdr)
dataf = fftshift(fft2(dataf, [zoom*m, zoom*k]))

fig = plt.figure(figsize=[16,5])
fig.add_subplot(1, 2, 1)
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[v_fold[1] * max_unambiguous_velocity,
                   (v_fold[1] +2 ) * max_unambiguous_velocity,
                    1.5*range_domain,
                   2.5*range_domain],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-40, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("second_focusing.png".format(number_target), dpi=300)


max_spec = np.max(data_fft2_db)
car2 = (data_fft2_db > (max_spec-threshold))
fig.add_subplot(1, 2, 2)
car_dilation = dilation(dilation(dilation(car2)))
plt.imshow(np.fliplr(car_dilation.T),
           aspect='auto',
           extent=[v_fold[1] * max_unambiguous_velocity,
                   (v_fold[1] +2 ) * max_unambiguous_velocity,
                    1.5*range_domain,
                    2.5*range_domain],
           interpolation='none',
           cmap='gray_r')
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("second_thresholding.png".format(number_target), dpi=300)


# %% de-transform
# est_a = deg2rad(0.5 * theta[0] + 0.5 * theta[1])

MARKER = "X"
MARKER_SIZE = 15
MARKER_COLOR = 'gray'

car = car2
i = 0
vre = v_est[1]

theta_ = np.mean(theta[0:2]) * np.ones((2,))
theta_ = theta_

indY, indX = np.nonzero(car)
image_y, image_x = car.shape
y = np.linspace(-0.5, 0.5, image_y, endpoint=False)[indY]

cof1_y = c*R[i]/fc/T/2/vre/np.tan(deg2rad(theta_[i])) * y
cof1_x = (indX - image_x/2) *resolution/zoom

# plt.figure()
# plt.scatter(cof1_y, cof1_x)

nX, nY = rotate_target(cof1_x, cof1_y, -theta_[i])

plt.figure()
plt.scatter(nY, nX, s=MARKER_SIZE, marker=MARKER, c=MARKER_COLOR)
plt.xlim(-2, 2)
plt.ylim(-2, 4)


car = car1
i = 1
vre = v_est[0]

indY, indX = np.nonzero(car)
image_y, image_x = car.shape
y = np.linspace(-0.5, 0.5, image_y, endpoint=False)[indY]

cof1_y = c*R[i]/fc/T/2/vre/np.tan(deg2rad(theta_[i])) * y
cof1_x = (indX - image_x/2) *resolution/zoom

# plt.figure()
# plt.scatter(cof1_y, cof1_x)

nX, nY = rotate_target(cof1_x, cof1_y, -theta_[i])

plt.figure()
plt.scatter(nY, nX, s=MARKER_SIZE, marker=MARKER, c=MARKER_COLOR)
plt.xlim(-2, 2)
plt.ylim(-1, 5)




plt.show()
