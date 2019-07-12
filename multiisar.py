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

# %% settings of simulation
save_fig = False


# %% setting basic parameters
k = 200  # fast time
m = 156  # slow time
SNR = -5
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
T = 3e-4
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
R = [14.619, 14.19, 10, 10]

number_target = 2
random = np.random.rand(4)  # add random velocity to the targets
v = (np.array([72, 80, 14, 14]) + 70) / 3.6 + random * 0.1 # velocity ()
a = np.array([0, 0, 10, 10]) + np.random.rand(4) * 1

theta = [20, 35, 30, 30]    # angle (should be similar)
w = [0, 0, 0, 0]           # rotational velocity
vr = v*cos(deg2rad(theta))  # radial velocity
ar = a*cos(deg2rad(theta))
print(vr)
vt = v*sin(deg2rad(theta))  # translational velocity
w = w + vt/R            # rotational velocity + translational_velocity / range

ele = 81                                    # number of the searching grids for acceleration
cle = 201                                   # number of the searching grids for velocity
vspan = np.linspace(np.min(vr[0:2])-2, np.max(vr[0:2])+2, cle)
ascan = np.linspace(-10, 10, ele)

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


Xcr, Ycr = rotate_target(Xc, Yc, theta[0])
low_alpha = 0.5      # variation of the amplitude (real)     # X:fast time
for i in range(number_scatters):
    data1 = data1 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j * pi * (fr * (R[0]+Xcr[i]) * K +  # range
                                    fd * (vr[0] + w[0]*Ycr[i]) * M +  # Doppler
                                    ar[0] * fa * K * M * M +  #
                                    ar[0] * frs * M * M +
                                    Cr[0] * K * M))
round_range1, _ = fold((R[0] + Ycr)*fr)
round_velocity1, fold1 = fold((vr[0] + w[0]*Xcr)*fd)


Xcr1, Ycr1 = rotate_target(Xc, Yc, theta[1])
for i in range(number_scatters):
    data2 = data2 + (low_alpha + (1-low_alpha) * np.random.rand()) * \
                    exp(-2j * pi * (fr * (R[1]+Xcr1[i]) * K +
                                    fd * (vr[1] + w[1]*Ycr1[i]) * M +
                                    ar[1] * fa * K * M * M +
                                    ar[1] * frs * M * M +
                                    Cr[1] * K * M))
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

data1f = fftshift(fft(data*(exp(0j * pi * (Cr[0] * K * M))) , axis=-1, n=4*k), axes=-1)
plt.figure(figsize=[8, 5])
plt.imshow(np.flipud(20 * log10(abs(data1f))).T, aspect='auto', cmap='jet', extent=[0, m, 1.5*range_domain, 2.5*range_domain])
plt.clim(vmin=22, vmax=62)
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
plt.xlim((-10, 10))
plt.ylim((0, 20))
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(axis='x', ls=":")
if save_fig:
    plt.savefig("scenario.png", dpi=300)
# plt.legend()

# %% 2DFFT
I = 0
dataf = fftshift((fft2((data)*exp(0j*pi*(Cr[I] * K * M + ar[I] * fa * K * M * M + ar[I] * frs * M * M)), [m, k])), -1)
plt.figure(figsize=[8, 5])
data_fft2_db = 20*log10(abs(dataf))
plt.imshow(np.flipud(data_fft2_db).T,
           aspect='auto',
           cmap='jet',
           extent=[-vm, vm, 0, 200*resolution],
           interpolation='none')
plt.clim(vmin=np.max(data_fft2_db)-20, vmax=np.max(data_fft2_db))
cbar = plt.colorbar()
cbar.set_label('dB', fontsize=15, rotation=-90, labelpad=18)
plt.ylabel('Range ($m$)')
plt.xlabel('Doppler ($m/s$)')
if save_fig:
    plt.savefig("2DFFT_{}.png".format(number_target), dpi=300)

# %% Using ME or VSVD to separate targets and estimate the couplings


cspan = vspan * fdr


def Fourier(cspan, data, X, Y, algorithm, alpha=1):
    ec = np.zeros((cle,), dtype=np.float32)
    tic = time.time()
    for i, com in (enumerate(cspan)):
        datac = data * exp(2j*pi*com*X*Y)
        isar = abs(fft2(datac, [1 * k, 1 * m]))
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
me1 = angle_acceleration_search(data, method1, ascan, cspan, K, M, algorithm=algorithm1, alpha=alpha)
me2 = angle_acceleration_search(data, method2, ascan, cspan, K, M, algorithm=algorithm2, alpha=alpha)
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



# %% fold Doppler
r0 = (R[0] + Xcr)
v0 = (vr[0] + w[0]*Ycr)
r1 = (R[1] + Xcr1)
v1 = (vr[1] + w[1]*Ycr1)
range_fold = R[0] // range_domain
doppler_fold = vr[0] // (2*max_unambiguous_velocity) *2

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
plt.ylim(1.5*range_domain, 2.5*range_domain)
plt.xlim(doppler_fold*max_unambiguous_velocity, (doppler_fold+2)*max_unambiguous_velocity)
if save_fig:
    plt.savefig("unfolded_velocity_{}.png".format(number_target), dpi=300)


plt.show()
