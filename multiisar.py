#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:58:07 2019

@author: shengzhixu
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from pylab import pi, exp, fft, fft2, log10, fftshift, sin, cos, deg2rad, rad2deg
from scipy.constants import speed_of_light as c
from tqdm import tqdm

from multi_isar.Keystone import Keystone
from multi_isar.utils import cart2pol, pol2cart, awgn, normalize, entropy, renyi, tsallis, image_constrast, db
# from multi_isar.detect_local_minima import detect_local_minima
from skimage.restoration import denoise_tv_chambolle, denoise_tv_chambolle
from skimage.morphology import extrema

#%%

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'figure.max_open_warning': 0}

plt.rcParams.update(params)

'''
Multi-targets ISAR separation comparison between

1) Minimum Entropy
2) Variance of the Singular values

Conclusions:
    1) Variance has better separation performance
    2) Variance cannot resolve higher order couplings
        for example:
            if the coupling has X*Y^2 and X*Y, they cannot be resolved correctly
            by svd as they located in the same singular vector
'''

#os.chdir('~/Documents/Python')
plt.close('all')

# %% settings of simulation
save_fig = False    # always keep it false


# %% setting basic parameters
k = 200  # fast time
m = 156  # slow time
l = 8    # elements
[K, M, L] = np.meshgrid(np.arange(k), np.arange(m), np.arange(l))  # slow time * fast time (X is fasttime, Y is slowtime)
# 156*200     X(0->200), Y(0->156)

# %%

car = np.load('/Users/shengzhixu/Documents/Python/multi_isar/car.npz')
dxy = 2
Xc = car['arr_0'][::dxy]
Yc = car['arr_1'][::dxy]
Xc = (Xc - np.max(Xc)//2) / np.max(Xc) * 5   /  1 - 0.45
Yc = (Yc - np.max(Yc)//2) / np.max(Yc) * 2.5 /  1 - 0.2

plt.scatter(Yc, Xc)
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
f0 = 77e9
fc = f0 + B/2   # carrier frequency
T = 4.0e-4
PRF = 1/T
Td = 0.8*T


Ts = Td/k
fs = 1/Ts
mu = B/Td
vm = c/4/T/fc    #(-vm, vm)

'''
a = g = 10 m/s
0.5*a*(T*M)**2 = 0.5 * 10 * (4e-4 * 156)**2 = 0.010951
dr = c/2/B = 3e8/2/4e9 = 0.0375

Acceleration rate can be ignored!!!!!!!!!????   No!!
'''

# %% Targets parameters
range_domain = resolution * k
R = [22.919, 21.60, 10, 10]
random = np.random.rand(4)                                    # add random velocity to the targets
v = -(np.array([90, 80, 14, 14]) + 80) / 3.6 + random * 0     # velocity ()
a = np.array([-4, 4, 10, 10])*0 + np.random.rand(4) * 0

# v = -(np.array([95, 80, 14, 14]) + 60 + 20 * np.random.randn()) / 3.6 + random * 1
v = np.array([-40.32, -40.36, -21.82, -21.82]) + random * 0.000005
# v = np.array([-41.2, -39.97, -21.37, -21.33]) + random * 0.1 # for the paper

v_kilo = v*3.6
v_self = 70
v_car = v_kilo + v_self

print('The velocity of car1 and car2 are: {:.2f}km/h and {:.2f}km/h'.format(v_car[0], v_car[1]))
print('The velocity of radar is: {:.2f}km/h'.format(v_self))

theta = [20, 30, 30, 30]                                      # angle (should be similar)
w = np.array([0, 0, 0, 0]) + 0.001 * np.random.randn(4)       # rotational velocity
vr = v*cos(deg2rad(theta))                                    # radial velocity
ar = a*cos(deg2rad(theta))
print(vr)
vt = v*sin(deg2rad(theta))                                    # translational velocity
w = w + vt/R                                                  # rotational velocity + translational_velocity / range


#%% special settings
ele = 21      # 41 for paper                              # number of the searching grids for acceleration
cle = 81      # 121 for paper                             # number of the searching grids for velocity

keystone_usd = 0
algorithm11 = renyi                   # eigen
algorithm22 = image_constrast                   # fourier
with_spread = 1                       # True for isar, False for cluster targets
SNR = 10                           #
alpha_all_one = 1                     # True to set alhpa to 1
low_alpha = 0.5                       # variation of the amplitude (real)     # X:fast time
number_target = 2                    #
threshold = 5                         # db, threshold to split focused target
denosing = 1                          # True for de-noising, False for not
weight = 0.2                         # De-noising weight
h = 0.05                               # peak prominences pf extrema


vspan = np.linspace(np.min(vr[0:number_target])-2, np.max(vr[0:number_target])+2, cle)
ascan = np.linspace(-4, 4, ele)

# %% Generating data
data1 = np.zeros((m, k, l), dtype=complex)
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
    remainder = value - np.floor(value + 0.5)
    fold_num = np.floor(value)
    return remainder, fold_num


Xcr, Ycr = rotate_target(Xc, Yc, theta[0])


def complex_amplitude(low_alpha):
    COF = np.sqrt(2)/2
    real = COF * (low_alpha + (1-low_alpha) * np.random.rand())
    imag = COF * (low_alpha + (1-low_alpha) * np.random.rand())

    return real+ 1j * imag


def varying_amplitude(X, low_alpha=0.5):
    k = X.size
    alpha = np.zeros_like(X, dtype=complex)
    for i in range(k):
        alpha[i] = complex_amplitude(low_alpha) * (np.random.randn() * 10 + 10)

    return alpha


def swerlingI(sigma):
    pass


# %%
alpha1 = varying_amplitude(Xc, low_alpha)
alpha2 = varying_amplitude(Xc, low_alpha)
# alpha2 = alpha1
# plt.figure()
# plt.plot(np.sort(np.abs(alpha1)))
# plt.plot(np.sort(np.abs(alpha2)))

# alpha2 = alpha1
# np.savez('alpha.npz', alpha1, alpha2)
#
alpha = np.load('alpha.npz')
alpha1 = alpha['arr_0']
alpha2 = alpha['arr_1']
#
# # plt.plot(np.sort(np.abs(alpha1)), ls=":")
# # plt.plot(np.sort(np.abs(alpha2)), ls=":")

if alpha_all_one:
    alpha1 = np.ones(500, ) * ( np.random.rand(500, )*0 + 1 )
    alpha2 = alpha1


fig = plt.figure(figsize=[8, 5])
binImg = np.load('multi_isar/data/binImg.npy')
ax = fig.add_subplot(1, 1, 1)
plt.imshow(binImg, cmap='gray_r')
plt.xlim(100, 1180)
plt.ylim([40, 470])
plt.scatter(car['arr_0'][::dxy], car['arr_1'][::dxy], s=100*abs(alpha2), edgecolors='r', marker='8')
ax.patch.set_visible(False)                                     # remove the frame
for spi in plt.gca().spines.values():                           # remove the frame
    spi.set_visible(False)
plt.tight_layout()
plt.axis('off')
if save_fig:
    plt.savefig('car_scattering_model.png', dpi=300)

# %% image the scene

def scene(Xc, Yc, theta, R):
    Xt, Yt = rotate_target(Xc, Yc, theta)   # Cart
    new_Xt = Xt + R                         # Cart
    Xtb, Ytb = rotate_target(new_Xt, Yt, -theta)    # Cart

    return Xtb, Ytb

if alpha_all_one:
    alpha_zoom = 5
else:
    alpha_zoom = 2


x1, y1 = scene(Xc, Yc, theta[0], R[0])
x2, y2 = scene(Xc, Yc, theta[1], R[1])

angle1 = -np.arctan(y1/x1)
angle2 = -np.arctan(y2/x2)

print('Car1 Average Angle: {}'.format(np.mean(rad2deg(angle1))))
print('Car2 Average Angle: {}'.format(np.mean(rad2deg(angle2))))

f = lambda x, y: rad2deg(np.arctan(y/x))

plt.figure(figsize=[5, 5])
plt.scatter(y1, x1, s=abs(alpha1)*alpha_zoom, label='car1', c='b', marker='x')
plt.scatter(y2, x2, s=abs(alpha2)*alpha_zoom, label='car2', c='g', marker='x')
plt.plot([0, -7.6], [0, 19.6], c='b', lw=2, ls=':', label='LOS')                # plot line of sight
plt.plot([0, -10], [0, 16.2], c='g', lw=2, ls=':', label='LOS')
plt.scatter([0], [0], s=300, marker="v")
plt.xlim((-15, 15))
plt.ylim((0, 30))
plt.xlabel("X' ($m$)")
plt.ylabel("Y' ($m$)")
plt.grid(axis='x', ls=":")
plt.legend()

if save_fig:
    plt.savefig("scenario.png", dpi=300)
# plt.legend()

# %% establish the data
'''
Add angle information
'''
wavelength = c / fc
d = wavelength / 2

def angle_with_time(x, y, v, t):
    return np.arctan((x + v*t)/y)

# L = np.arange(l)
# steering_vector1 = exp(2j*pi*L*d/wavelength*sin(angle1))
# steering_vector2 = exp(2j*pi*L*d/wavelength*sin(angle2))

# %%

for i in range(number_scatters):
    data1 = data1 + alpha1[i] * \
                    exp(-2j * pi * (fr * (R[0]+Xcr[i]) * K +                    # range
                                    fd * (vr[0] + w[0]*Ycr[i]) * M +            # Doppler
                                    ar[0] * fa * K * M * M +
                                    ar[0] * frs * M * M +
                                    (vr[0] + with_spread * w[0] * Ycr[i]) * fdr * K * M
                                    + L*d/c*(f0 + mu*K*Ts)*sin(angle_with_time(x1[i], y1[i], vr[0], K))))
round_range1, _ = fold((R[0] + Ycr)*fr)
round_velocity1, fold1 = fold((vr[0] + w[0]*Xcr)*fd)


Xcr1, Ycr1 = rotate_target(Xc, Yc, theta[1])
for i in range(number_scatters):
    data2 = data2 + alpha2[i] * \
                    exp(-2j * pi * (fr * (R[1]+Xcr1[i]) * K +
                                    fd * (vr[1] + w[1]*Ycr1[i]) * M +
                                    ar[1] * fa * K * M * M +
                                    ar[1] * frs * M * M +
                                    (vr[1] + with_spread*w[1]*Ycr[i]) * fdr * K * M
                                    + L*d/c*(f0 + mu*K*Ts)*sin(angle_with_time(x1[i], y1[i], vr[1], K))))
round_range2, _= fold((R[1] + Ycr1)*fr)
round_velocity2, fold2 = fold((vr[1] + w[1]*Xcr1)*fd)


if number_target == 2:
    data = data1 + data2
else:
    data = data1

data_nonoise = data
sig_power = db(np.sum(abs(data_nonoise)**2))
data_noise = awgn(data, SNR)
# data \in C^{ m*k }

data = data_noise[..., 0]
# %% 1DFFT
[K, M] = np.meshgrid(np.arange(k), np.arange(m))
data1f = fftshift(fft(data*(exp(0*2j * pi * (Cr[0] * K * M))), axis=-1, n=4*k), axes=-1)
plt.figure(figsize=[8, 5])
data1f_db = 20 * log10(abs(data1f))
plt.imshow((data1f_db).T,
           aspect='auto',
           cmap='jet',
           extent=[0, m, 2.5*range_domain, 3.5*range_domain])
plt.clim(vmin=np.max(data1f_db) - 40, vmax=np.max(data1f_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Slow Time Bins')
if save_fig:
    plt.savefig("1DFFT_{}.png".format(number_target), dpi=300)



# %% 2DFFT
I = 1
# vr[I] = 36
comdata = data*exp(0*2j*pi*(vr[I]*fdr*K*M +ar[I]*fa*K*M*M + ar[I]*frs*M*M + vr[I]*fd*M))
# comdata = Keystone(comdata, fd, fdr)
dataf = fftshift((fft2(comdata, [4*m, 4*k])))
plt.figure(figsize=[8, 5])
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[-vm, vm, 2.5*range_domain, 3.5*range_domain],
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
def Fourier(cspan, data, K, M, algorithm, alpha=1, zoom=1):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, v in (enumerate(cspan)):
        datac = data * exp(2j * pi * (v * fdr * K * M + v * fd * M))
        if keystone_usd:
            datac = Keystone(datac, fd, fdr, verbose=False)
        isar = abs(fft2(datac, [zoom * k, zoom * m]))
        ec[i] = algorithm(isar**2, alpha)
        # ec[i] = -np.var(isar**2)
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
        eigvals = abs(np.linalg.eigvalsh(datac.T.conj().dot(datac)))
        es[i] = algorithm(eigvals, alpha)
        # es[i] = -np.var(eigvals)
    print("Time for EIG: {:.3f} seconds".format(time.time()-tic))

    indc = np.argwhere(np.min(es) == es).flatten()[0]
    cvalue = cspan[indc]

    return es, cvalue


def angle_acceleration_search(data, method, ascan, cspan, K, M, algorithm, alpha=1):
    me = np.zeros((ele, cle))
    for i, a in tqdm(enumerate(ascan)):     # for acceleration
        if method.__name__ is 'eigen':
            # a = 0
            datac = data * exp(2j * pi * (a * fa * K * M * M + 0 * frs * M * M))
        elif method.__name__ is 'Fourier':
            # a = a
            datac = data * exp(2j * pi * (a * fa * K * M * M + a * frs * M * M))
        else:
            print("Please Confirm your Method!")
            raise ValueError
        me[i, :] = method(cspan, datac, K, M, algorithm, alpha)[0]

    return me

# %% main


me0, me1, me2, me3 = None, None, None, None
method1, method2, method3, method0 = None, None, None, None
algorithm1, algorithm2, algorithm3, algorithm0 = None, None, None, None


method1 = eigen
algorithm1 = algorithm11

method2 = Fourier
algorithm2 = algorithm22

alpha = 1

me1 = angle_acceleration_search(data, method1, ascan, cspan, K, M, algorithm=algorithm1, alpha=alpha)
me2 = angle_acceleration_search(data, method2, ascan, cspan, K, M, algorithm=algorithm2, alpha=alpha)
if algorithm1 == image_constrast:
    me1 = - me1
if algorithm2 == image_constrast:
    me2 = - me2
# %% plt
CMAP = 'hot'


if me1 is not None:
    plt.figure(figsize=[8, 5])
    plt.imshow(np.flipud(normalize(me1[:, :])), aspect='auto', cmap=CMAP,
               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])

    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    cbar = plt.colorbar()
    cbar.set_label('Normalized Entropy', fontsize=15, rotation=-90, labelpad=18)
    plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
    plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    if save_fig:
        plt.savefig(method1.__name__ + algorithm1.__name__ + '{}_{}.png'.format(SNR, number_target), dpi=300)
    # plt.title('Eig with Entropy')

if me2 is not None:
    plt.figure(figsize=[8, 5])
    plt.imshow(np.flipud(normalize(me2)), aspect='auto', cmap=CMAP,
               extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    # plt.contour(-np.flipud(normalize(me2)), aspect='auto', cmap='jet', extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
    plt.xlabel("Velocity ($m/s$)")
    plt.ylabel("Acceleration ($m/s^2$)")
    cbar = plt.colorbar()
    cbar.set_label('Normalized Entropy', fontsize=15, rotation=-90, labelpad=18)

    plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
    plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8,  ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    if save_fig:
        plt.savefig(method2.__name__ + algorithm2.__name__ + '{}_{}.png'.format(SNR, number_target), dpi=300)
    # plt.title('Fourier with Entropy')


#%% COMBINE

if (me2 is not None) and (me1 is not None):
    me_com = -(1-normalize(me1)) * (1-normalize(me2))
    plt.figure(figsize=[8, 5])
    plt.imshow(np.flipud(me_com),
               aspect='auto', cmap=CMAP, extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
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
    plt.title('EigEntropy + FourierEntropy')



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
doppler_fold = vr[0] // (2*vm) * 2

fig = plt.figure(figsize=[6.5, 5])
# size = fig.get_size_inches()*fig.dpi          # Get current figure size
plt.scatter(v0 % (2*vm)-vm, r0,  marker='x', c='b', label="car1", s=4*alpha_zoom*abs(alpha1))
plt.scatter(v1 % (2*vm)-vm, r1, marker='x', c='g',  label="car2", s=4*alpha_zoom*abs(alpha1))
plt.legend(loc="best")
plt.xlabel("Folded Doppler velocity ($m/s$)")
plt.ylabel("Range (m)")
plt.ylim((2.5)*range_domain, (3.5)*range_domain)
plt.xlim(-vm, 1*vm)
plt.grid(linestyle=':')
if save_fig:
    plt.savefig("folded_velocity_{}.png".format(number_target), dpi=300)

# %% unfold
plt.figure(figsize=[6.5, 5])
plt.scatter(v0, r0, marker='x', c='b', label="Car 1", s=4*alpha_zoom*abs(alpha1))
plt.scatter(v1, r1, marker='x', c='g',  label="Car 2", s=4*alpha_zoom*abs(alpha2))

plt.xlabel("Doppler velocity ($m/s$)")
plt.ylabel("Range (m)")
plt.ylim(2.5*range_domain, 3.5*range_domain)
# plt.xlim((doppler_fold-1)*vm, (doppler_fold+3)*vm)
plt.xlim(-40.8, -32.4)
plt.vlines(x = (doppler_fold-2)*vm, ymin=0, ymax=100, color='r', linestyles=':', label='$v_m$')
plt.vlines(x = (doppler_fold)*vm, ymin=0, ymax=100, color='r', linestyles=':')
plt.vlines(x = (doppler_fold+2)*vm, ymin=0, ymax=100, color='r', linestyles=':')
plt.vlines(x = (doppler_fold+4)*vm, ymin=0, ymax=100, color='r', linestyles=':')
plt.vlines(x = (doppler_fold+6)*vm, ymin=0, ymax=100, color='r', linestyles=':')

plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=0, ymax=100, colors='b', lw=2.5, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=0, ymax=100, colors='b', lw=2.5, linestyle='-.')
plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=0, ymax=100, colors='g', lw=2.5, linestyle='-.')
plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=0, ymax=100, colors='g', lw=2.5, linestyle='-.')

plt.legend(loc="upper right")

if save_fig:
    plt.savefig("unfolded_velocity_{}.png".format(number_target), dpi=300)


# %% Estimator
CMAP = 'jet_r'
scatter_c = 'w'
MARKER = 'X'


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharex='all', sharey='all')

np.set_printoptions(precision=2)

me2n = np.flipud(normalize(me2))
me1n = np.flipud(normalize(me1))
me_com = np.flipud(denoise_tv_chambolle(normalize(-(1-normalize(me1))*(1-normalize(me2))), weight=weight))
me_comn = np.flipud(normalize(-denoise_tv_chambolle(1-normalize(me1), weight=weight)
                              *denoise_tv_chambolle(1-normalize(me2), weight=weight)))

ax[0, 0].imshow(me1n, aspect='auto', cmap=CMAP)
ax[0, 0].axis('off')
if denosing:
    me1n = denoise_tv_chambolle(me1n, weight=weight)

ax[1, 0].imshow(me1n, aspect='auto', cmap=CMAP)
ax[1, 0].axis('off')


arr1 = extrema.h_minima((me1n), h)
indy, indx = np.nonzero(arr1)
to_delete_list = []
number_target_est = indx.size
for i in range(number_target_est):
    if me1n[indy[i], indx[i]] >= 0.4:
        to_delete_list.append(i)
indx1 = np.delete(indx, to_delete_list)
indy1 = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx1], ascan[indy1]))
ax[1, 0].scatter(indx1, indy1, s=20, c=scatter_c, marker=MARKER)


ax[0, 1].imshow(me2n, aspect='auto', cmap=CMAP)
ax[0, 1].axis('off')

if denosing:
    me2n = denoise_tv_chambolle(me2n, weight=weight)
ax[1, 1].imshow((me2n), aspect='auto', cmap=CMAP)
ax[1, 1].axis('off')

arr2 = extrema.h_minima((me2n), h)
indy, indx = np.nonzero(arr2)
to_delete_list = []
number_target_est = indx.size

for i in range(number_target_est):
    if me2n[indy[i], indx[i]] >= 0.4:
        to_delete_list.append(i)
indx2 = np.delete(indx, to_delete_list)
indy2 = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx2], ascan[indy2]))
ax[1, 1].scatter(indx2, indy2, s=20, c=scatter_c, marker=MARKER)


ax[0, 2].imshow(me_com, aspect='auto', cmap=CMAP)
ax[0, 2].axis('off')
ax[1, 2].imshow(denoise_tv_chambolle(me_comn, weight=weight), aspect='auto', cmap=CMAP)
ax[1, 2].axis('off')


arr3 = extrema.h_minima(denoise_tv_chambolle(me_comn, weight=weight), h)
indy, indx = np.nonzero(arr3)
to_delete_list = []
number_target_est = indx.size

for i in range(number_target_est):
    if me_comn[indy[i], indx[i]] >= 0.4:
        to_delete_list.append(i)
indx3 = np.delete(indx, to_delete_list)
indy3 = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx3], ascan[indy3]))
ax[1, 2].scatter(indx3, indy3, s=20, c=scatter_c, marker=MARKER)

plt.tight_layout()

# temp arr4 without denoising
arr4 = extrema.h_minima((me_com), h)
indy, indx = np.nonzero(arr4)
to_delete_list = []
number_target_est = indx.size
for i in range(number_target_est):
    if me_comn[indy[i], indx[i]] >= 0.4:
        to_delete_list.append(i)
indx4 = np.delete(indx, to_delete_list)
indy4 = np.delete(indy, to_delete_list)
# if save_fig:
#     plt.savefig("denoising_estimation.png", dpi=400)

# %% plot for paper
CMAP = 'hot'


plt.figure(figsize=[8, 5])
plt.imshow(normalize(denoise_tv_chambolle(me1n, weight=weight)), aspect='auto', cmap=CMAP,
           extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
cbar = plt.colorbar()
cbar.set_label('Normalized Entropy', fontsize=15, rotation=-90, labelpad=18)
plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
if number_target == 2:
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
plt.scatter(vspan[indx1], -ascan[indy1], s=60, c=scatter_c, marker=MARKER)
plt.xlim(vspan[0], vspan[-1])
plt.ylim(ascan[0], ascan[-1])
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
if save_fig:
    plt.savefig("eigen_denoising_estimation.png", dpi=300)


plt.figure(figsize=[8, 5])
plt.imshow(normalize(denoise_tv_chambolle(me2n, weight=weight)), aspect='auto', cmap=CMAP,
           extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
cbar = plt.colorbar()
cbar.set_label('Normalized Entropy', fontsize=15, rotation=-90, labelpad=18)
plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
if number_target == 2:
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
plt.scatter(vspan[indx2], -ascan[indy2], s=60, c=scatter_c, marker=MARKER)
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
if save_fig:
    plt.savefig("fourier_denoising_estimation.png", dpi=300)


plt.figure(figsize=[8, 5])
plt.imshow(normalize(denoise_tv_chambolle(me_comn, weight=weight)), aspect='auto', cmap=CMAP,
           extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
cbar = plt.colorbar()
cbar.set_label('Normalized Entropy', fontsize=15, rotation=-90, labelpad=18)
plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
if number_target == 2:
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
plt.scatter(vspan[indx3], -ascan[indy3], s=60, c=scatter_c, marker=MARKER)
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
if save_fig:
    plt.savefig("combination_denoising_estimation.png", dpi=300)


plt.figure(figsize=[8, 5])
plt.imshow(normalize((me_com)), aspect='auto', cmap=CMAP,
           extent=[vspan[0], vspan[-1], ascan[0], ascan[-1]])
cbar = plt.colorbar()
cbar.set_label('Normalized Entropy', fontsize=15, rotation=-90, labelpad=18)
plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='b', lw=3, linestyle='-.')
if number_target == 2:
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=ascan[0], ymax=ascan[-1], colors='g', lw=3, linestyle='-.')
plt.scatter(vspan[indx4], -ascan[indy4], s=60, c=scatter_c, marker=MARKER)
plt.xlabel("Velocity ($m/s$)")
plt.ylabel("Acceleration ($m/s^2$)")
if save_fig:
    plt.savefig("combination_noising_estimation.png", dpi=300)

# %% estimation of parameters

v_est = vspan[indx3]
a_est = -ascan[indy3] * 1
v_fold = v_est // (2*vm) * 2


# %% thresholding
def triple_dilation(X):
    return dilation(dilation(dilation(dilation(dilation(X)))))

zoom = 4
dataf = (data)* exp(2j*pi*(v_est[0]*fdr*K*M + a_est[0]*fa*K*M*M + a_est[0]*frs*M*M + v_est[0]*fd * M))
# dataf = data* exp(2j*pi*(v_fold[0]*vm*fdr*K*M + a_est[0]*fa*K*M*M + a_est[0]*frs*M*M))
if with_spread:
    dataf = Keystone(dataf, fd, fdr)
dataf = fftshift(fft2(dataf, [zoom*m, zoom*k]))

car1_color_offset = np.ones(shape=[*dataf.T.shape, 3])
car2_color_offset = np.ones_like(car1_color_offset)

plt.figure(figsize=[8, 5])
data_fft2_db = 20*log10(abs(dataf))
plt.imshow((np.flipud(data_fft2_db).T),
           aspect='auto',
           cmap='jet',
           extent=[ -vm + v_est[0],
                    vm + v_est[0],
                    2.5*range_domain,
                   3.5*range_domain],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-30, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('(dB)', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("first_focusing.png".format(number_target), dpi=300)

from skimage import morphology
dilation = morphology.binary_dilation
max_spec = np.max(data_fft2_db)

car1 = (data_fft2_db > (max_spec-threshold))
car1_dilation = triple_dilation(car1)
car1_color_offset[:, :, 0] = 1 - np.fliplr(car1_dilation.T)
car1_color_offset[:, :, 1] = 1 - np.fliplr(car1_dilation.T)
plt.figure(figsize=[6.5, 5])
plt.imshow(car1_color_offset,
           aspect='auto',
           extent=[-vm + v_est[0],
                    vm + v_est[0],
                    2.5*range_domain,
                   3.5*range_domain],
           interpolation='none',
           cmap='gray_r')
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
plt.grid(ls=':')
if save_fig:
    plt.savefig("first_thresholding.png".format(number_target), dpi=300)

# data_remove_car1 = ifft(ifftshift(dataf * (~dilation(car1))), [m,k])
# data_remove_car1 = data_remove_car1 * exp(-2j*pi*(v_est[0]*fdr*K*M + a_est[0]*fa*K*M*M + a_est[0]*frs*M*M + v_est[0]*fd * M))


if v_est.size >= 2:
    #
    # data = data_remove_car1
    #
    dataf = (data)*exp(2j*pi*(v_est[1]*fdr*K*M+a_est[1]*fa*K*M*M+a_est[1]*frs*M*M+ v_est[1]*fd * M))
    if with_spread:
        dataf = Keystone(dataf, fd, fdr)
    dataf = fftshift(fft2(dataf, [zoom*m, zoom*k]))

    plt.figure(figsize=[8, 5])
    data_fft2_db2 = 20*log10(abs(dataf))
    plt.imshow((np.flipud(data_fft2_db2).T),
               aspect='auto',
               cmap='jet',
               extent=[-vm + v_est[1],
                        vm + v_est[1],
                        2.5*range_domain,
                        3.5*range_domain],
                        interpolation='none')
    max_data_fft2_db2 = np.max(data_fft2_db2)
    plt.clim(vmin=max_data_fft2_db2-30, vmax=max_data_fft2_db2)
    cbar = plt.colorbar()
    cbar.set_label('(dB)', fontsize=15, rotation=-90, labelpad=18)
    plt.ylabel('Range ($m$)')
    plt.xlabel('Doppler ($m/s$)')
    if save_fig:
        plt.savefig("second_focusing.png".format(number_target), dpi=300)

    max_spec = np.max(data_fft2_db2)
    car2 = (data_fft2_db2 > (max_spec-threshold))
    car2_dilation = triple_dilation(car2)
    car2_color_offset[:, :, 0] = 1 - np.fliplr(car2_dilation.T)
    car2_color_offset[:, :, 1] = 1 - 0.5 * np.fliplr(car2_dilation.T)
    car2_color_offset[:, :, 2] = 1 - np.fliplr(car2_dilation.T)
    plt.figure(figsize=[6.5, 5])
    plt.imshow(car2_color_offset,
               aspect='auto',
               extent=[-vm + v_est[1],
                        vm + v_est[1],
                        2.5*range_domain,
                        3.5*range_domain],
               interpolation='none')
    plt.ylabel('Range ($m$)')
    plt.xlabel('Doppler ($m/s$)')
    plt.grid(ls=':')
    if save_fig:
        plt.savefig("second_thresholding.png".format(number_target), dpi=300)


#%%
plt.figure(figsize=[6.5, 5])
if v_est.size >= 2:
    plt.imshow(np.fliplr(car2_dilation.T),
               aspect='auto',
               extent=[-vm + v_est[1],
                       vm + v_est[1],
                       2.5 * range_domain,
                       3.5 * range_domain],
               interpolation='none',
               cmap='gray_r')
plt.imshow(np.fliplr(car1_dilation.T),
           aspect='auto',
           extent=[ -vm + v_est[0],
                    vm + v_est[0],
                    2.5*range_domain,
                    3.5*range_domain],
           interpolation='none',
           cmap='gray_r')


plt.xlabel("Doppler velocity ($m/s$)")
plt.ylabel("Range (m)")
plt.ylim(2.5*range_domain, 3.5*range_domain)
plt.xlim(-40.8, -32.4)
plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=0, ymax=100, colors='b', lw=2.5, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=0, ymax=100, colors='b', lw=2.5, linestyle='-.')
if v_est.size >= 2:
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=0, ymax=100, colors='g', lw=2.5, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=0, ymax=100, colors='g', lw=2.5, linestyle='-.')

plt.vlines(x = (doppler_fold-2)*vm, ymin=0, ymax=100, color='r', linestyles=':', label='$v_m$')
plt.vlines(x = doppler_fold * vm, ymin=0, ymax=100, color='r', linestyles=':')
plt.vlines(x = (doppler_fold+2)*vm, ymin=0, ymax=100, color='r', linestyles=':')
plt.vlines(x = (doppler_fold+4)*vm, ymin=0, ymax=100, color='r', linestyles=':')
plt.vlines(x = (doppler_fold+6)*vm, ymin=0, ymax=100, color='r', linestyles=':')

if save_fig:
    plt.savefig("map_recover.png", dpi=300)

#%% velocity comparison
plt.figure(figsize=[8, 5])
plt.plot(vspan, normalize(me1[ele//2, :]), lw=4, label='Normalized EES', c='r', ls=":")
plt.plot(vspan, normalize(me2[ele//2, :]), lw=4, label='Normalized EFS', c='c', ls=":")
plt.plot(vspan, normalize(me_com[ele//2, :]), lw=4, label='Normalized Combination', c='m', ls=":")

if denosing:
    plt.plot(vspan, normalize(me1n[ele // 2, :]), lw=3, label='Denoised EES', c='r')
    plt.plot(vspan, normalize(me2n[ele // 2, :]), lw=3, label='Denoised EFS', c='c')
    plt.plot(vspan, normalize(me_comn[ele // 2, :]), lw=3, label='Denoised Combination', c='m')

plt.grid(ls=":")
plt.legend(prop={'size': 10})
plt.vlines(x=vr[0] + w[0] * np.min(Xcr)*0.8, ymin=-0.2, ymax=1.3, colors='b', lw=2, linestyle='-.')
plt.vlines(x=vr[0] + w[0] * np.max(Xcr)*0.8, ymin=-0.2, ymax=1.3, colors='b', lw=2, linestyle='-.')
if v_est.size >= 2:
    plt.vlines(x=vr[1] + w[1] * np.min(Xcr1)*0.8, ymin=-0.2, ymax=1.3, colors='g', lw=2, linestyle='-.')
    plt.vlines(x=vr[1] + w[1] * np.max(Xcr1)*0.8, ymin=-0.2, ymax=1.3, colors='g', lw=2, linestyle='-.')
plt.xlabel("velocity $(m/s)$")
plt.ylabel("Normalized Entropy")
plt.ylim(-0.2, 1)
plt.gca().invert_yaxis()
if save_fig:
    plt.savefig("velocity_comparison.png", dpi=300)

# %% GO BACK TO ESTIMATE THE ANGLE OF EACH CAR
v_window = np.linspace(vm, -vm, data_fft2_db.shape[0], endpoint=False)
r_window = np.linspace(0, range_domain, data_fft2_db.shape[1], endpoint=False)


car1_th = data_fft2_db * (data_fft2_db > (data_fft2_db - threshold))
car_arr1 = extrema.h_minima(-car1_th, h=15)
indv1, indr1  = np.nonzero(car_arr1)
# plt.figure()
# plt.scatter(indv1, indr1)

v1_estimate = v_est[0] + v_window[indv1]
r1_estimate = r_window[data_fft2_db.shape[1] - indr1] - 0.5*range_domain
# plt.figure()
# plt.plot(v1_estimate)
# plt.plot(vr[0] + w[0]*Ycr)

car2_th = data_fft2_db2 * (data_fft2_db2 > (data_fft2_db2 - threshold))
car_arr2 = extrema.h_minima(-car2_th, h=15)
# plt.imshow(data_fft2_db)
indv2, indr2 = np.nonzero(car_arr2)
# plt.figure()
# plt.scatter(indv2, indr2)

v2_estimate = v_est[1] + v_window[indv2]
r2_estimate = r_window[data_fft2_db.shape[1] - indr2] - 0.5*range_domain
# plt.figure()
# plt.plot(v2_estimate)
# plt.plot(np.sort(vr[1] + w[1]*Ycr))

plt.figure()
plt.scatter(v1_estimate, r1_estimate)
plt.scatter(v2_estimate, r2_estimate)
plt.grid(ls=':')

# %% Beam-forming for the whole data
[K, M, L] = np.meshgrid(np.arange(k), np.arange(m), np.arange(l))
data_car1 = np.zeros_like(K, dtype=complex)
data_car2 = np.zeros_like(data_car1)


for j in range(v1_estimate.size):
    data_car1 = data_car1 + exp(2j * pi * (fr * r1_estimate[j] * K +
                                            fd * v1_estimate[j] * M))

for j in range(v2_estimate.size):
    data_car2 = data_car2 + exp(2j * pi * (fr * r2_estimate[j] * K +
                                            fd * v2_estimate[j] * M ))

angle_scan_length = 181
bf_scan = np.linspace(-45, 45, angle_scan_length)
bf1 = np.zeros((bf_scan.size, ), dtype=complex)
bf2 = np.zeros((bf_scan.size, ), dtype=complex)
bf3 = np.zeros((bf_scan.size, ), dtype=complex)
print("starting angle beam-forming")
for i, phi in tqdm(enumerate(bf_scan)):
    steering_vector = exp(2j * pi * L * d / wavelength * sin(deg2rad(phi)))
    bf3[i] = db(abs(np.sum(steering_vector[:, :, :] * data_noise[:, :, :])))
    bf1[i] = db(abs(np.sum((steering_vector*data_car1*data1*exp(2j*pi*v_est[0]*fdr*K*M))[:, :, :])))
    bf2[i] = db(abs(np.sum((steering_vector*data_car2*data2*exp(2j*pi*v_est[1]*fdr*K*M))[:, :, :])))

plt.figure()
plt.plot(bf_scan, normalize(bf1, mode='80'), label='car1')
plt.plot(bf_scan, normalize(bf2, mode='80'), label='car2')
plt.plot(bf_scan, normalize(bf3, mode='80'), label='whole')
plt.grid(ls=':')
plt.xlabel('Angle (degree)')
plt.ylabel('Amplitude (dB')
plt.legend()
if save_fig:
    plt.savefig('SeparatedBeamForming.png', dpi=300)

angle_est1 = bf_scan[np.argmax(bf1)]
angle_est2 = bf_scan[np.argmax(bf2)]

print(angle_est1, angle_est2)

# %% de-transform from thresholding results
MARKER = "X"
MARKER_SIZE = 15
MARKER_COLOR = 'gray'

v1_ori = v_est[0] / cos(deg2rad(angle_est1))
v2_ori = v_est[1] / cos(deg2rad(angle_est2))

w1_ori = v1_ori * sin(deg2rad(angle_est1)) / np.mean(r1_estimate + range_fold*range_domain)
w2_ori = v2_ori * sin(deg2rad(angle_est2)) / np.mean(r2_estimate + range_fold*range_domain)

# %%
from skimage.transform import rotate, resize

# resize -> rotation ->
resize_car2 = resize(car2_dilation, (int(car2_dilation.shape[0]/-w2_ori), car2_dilation.shape[1]), clip=False)
rotated_car2 = rotate(resize_car2, -angle_est2)
plt.figure()
plt.imshow(car2_dilation)
plt.figure()
plt.imshow(rotated_car2)
plt.title('Resize Car2')

resize_car1 = resize(car1_dilation, (int(car1_dilation.shape[0]/-w1_ori), car1_dilation.shape[1]), clip=False)
rotated_car1 = rotate(resize_car1, -angle_est1)
plt.figure()
plt.imshow(car1_dilation)
plt.figure()
plt.imshow(rotated_car1)
plt.title('Resize Car1')

lenX1, lenY1 = rotated_car1.shape
lenX2, lenY2 = resize_car2.shape
# %%
# image the scene
X1, Y1 = (np.mean(r1_estimate)*0 + range_fold*range_domain) \
         * np.array([-sin(deg2rad(angle_est1)), cos(deg2rad(angle_est1))])
X2, Y2 = (np.mean(r2_estimate)*0 + range_fold*range_domain ) \
         * np.array([-sin(deg2rad(angle_est2)), cos(deg2rad(angle_est2))])


range_resolution_zoom = resolution / zoom
X1_g = int(X1//range_resolution_zoom)
Y1_g = int(Y1//range_resolution_zoom)
X2_g = int(X2//range_resolution_zoom)
Y2_g = int(Y2//range_resolution_zoom)
grid_X = int(30/range_resolution_zoom)
image_sc1 = np.zeros((grid_X, grid_X), dtype=bool)
image_sc2 = np.zeros((grid_X, grid_X), dtype=bool)

image_sc1[Y1_g-lenY1//2:Y1_g + lenY1//2, grid_X//2 + X1_g - lenX1//2:grid_X//2 + X1_g + lenX1//2 + int(lenX1%2==1)] \
                            = np.flipud(rotated_car1.T)
image_sc2[Y2_g-lenY2//2:Y2_g + lenY2//2, grid_X//2 + X2_g - lenX2//2:grid_X//2 + X2_g + lenX2//2 + int(lenX2%2==1)] \
                            = np.flipud(rotated_car2.T)
plt.figure()
plt.imshow(np.flipud(image_sc1 + image_sc2),
           extent=[-15, 15, 0, 30])
plt.set_cmap('gray_r')
plt.grid(axis='x', ls=':')
if save_fig:
    plt.savefig('Recovered_Geometry.png', dpi=300)


# %%
plt.show()
