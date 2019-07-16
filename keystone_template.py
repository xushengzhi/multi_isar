# -*- coding: utf-8 -*-
'''
Creat on 2019-04-17

Authors: shengzhixu

Email: sz.xu@hotmail.com

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab  import pi, log2, floor, ceil, zeros, exp, sinc, fft, fftshift, log10, angle, ifft, ifftshift
from scipy.constants import speed_of_light as c
from scipy import interpolate

from Keystone import sinc_interp

# %% utils

def mmplot(mat, clim=40):
    '''
    db plot
    :param mat: input matrix
    :param clim: axis clim
    :return: None, plt object
    '''
    mat = 20*log10(abs(mat))
    mat = mat - np.max(mat)
    plt.imshow(mat, cmap='jet', aspect='auto')
    plt.clim([-clim, 0])
    plt.colorbar()

def interp(xp, xt, x):
    """
    Interpolate the signal to the new points using a sinc kernel
    input:
    xt    time points x is defined on
    x     input signal column vector or matrix, with a signal in each row
    xp    points to evaluate the new signal on
    output:
    y     the interpolated signal at points xp
    """

    mn = x.shape
    if len(mn) == 2:
        m = mn[0]
        n = mn[1]
    elif len(mn) == 1:
        m = 1
        n = mn[0]
    else:
        raise ValueError ("x is greater than 2D")

    nn = len(xp)

    y_real = np.zeros((m, nn))
    y_imag = np.zeros((m, nn))

    for (pi, p) in enumerate(xp.real):
        si = np.tile(np.sinc (xt - p), (m, 1))
        y_real[:, pi] = np.sum(si * x)
    for (pi, p) in enumerate(xp.imag):
        si = np.tile(np.sinc (xt - p), (m, 1))
        y_imag[:, pi] = np.sum(si * x)

    return (y_real + 1j*y_imag).squeeze()

# %% USER INPUT SECTION
L = 128  # fast time dimension, samples
M = 101  # slow time dimension, samples,  keep it odd

# K_L and K_M must be even to avoid labeling problems later
K_L = int(2**(ceil(log2(L))+1))  # fast time DFT size for interpolation and shifting
K_M = int(2**(ceil(log2(M))+3))  # slow time DFT size

Ntgt = 3
v = np.array([-200, -60, 650]) # velocity in m/s towards the radar

init_range_bins = np.array([30, 60, 65])
carrier_frequency = 10e9 # RF
B = 200e6 # bandwidth
# sampling intervals and rates
sampling_frequency = 2.3 * B
PRF = 10e3
# order of sinc interpolating filter
Nsinc = 11

# %% DERIVED PARAMETERS
m_end = (M-1)/2
ms = np.arange(M) - m_end
Fd = 2 * v * carrier_frequency / c
sampling_interval = 1 / sampling_frequency
dr = c * sampling_interval / 2  # range bin spacing
T = 1 / PRF
Dfd = 1/M
Drb = (1/B) / sampling_interval
if any(PRF < Fd/2):
    print('Warning: PRF < Fd/2')
    # raise ValueError

# computeand report total range migration over the CPI in range bins
RM = v * T / dr # amount of range migration per pulse in range bins
RMtot = M*RM

fd = Fd * T
amb_num , fdn = divmod(fd+0.5, 1) # fold number
fdn = fdn - 0.5

maximum_unambiguous_velocity = c/4/T/carrier_frequency

L1 = init_range_bins - Drb
L2 = init_range_bins + Drb
fd1 = fdn - Dfd
fd2 = fdn + Dfd

# %% CREATE SYNTHETIC DATA
y = zeros((L, M), dtype=complex)    # ft
for i in range(Ntgt):
    del_phi = -4 * pi * (carrier_frequency / c) * v[i] * T
    for m in range(M):
        mm = ms[m]
        y[:, m] = y[:, m] + exp(-1j*del_phi*mm) * \
                  sinc(B * sampling_interval * (np.arange(L) - init_range_bins[i] + v[i] * T * mm / dr))

plt.figure()
plt.imshow(abs(y), aspect='auto', cmap='gray_r')
plt.grid()
plt.xlabel('pulse number')
plt.ylabel('range bin')
plt.title('Raw Fast-time/Slow-time Data pattern')

Y_rd = fftshift( fft(y, n=K_M, axis=-1), axes=-1)   # ff (sinc pattern)
plt.figure()
mmplot(Y_rd)
plt.xlabel('normalized Doppler')
plt.ylabel('range bin')
plt.title('Raw Range-Doppler Matrix')

# %%
Y_Rd = fftshift(fft(y, K_L, axis=0), axes=0)      # tt (sinusoidal pattern)
FL = (np.arange(K_L) - K_L/2) / K_L * sampling_frequency
fig = plt.figure()
fig.add_subplot(121)
plt.imshow(abs(Y_Rd))
plt.colorbar()
plt.xlabel('slow time')
plt.ylabel('fast time frequency')
plt.title('Magnitude')
fig.add_subplot(122)
plt.imshow(angle(Y_Rd))
plt.colorbar()
plt.xlabel('slow time')
plt.ylabel('fast time frequency')
plt.title('Unwrapped phase')

# %% Keystone interpolation
Y_Rd_key = np.zeros_like(Y_Rd, dtype=complex)
for k in range(K_L):
    y_temp = sinc_interp(Y_Rd[k, :],
                         ms,
                         ms * (carrier_frequency / (carrier_frequency + FL[k])))
    Y_Rd_key[k, :] = y_temp

# for mp in range(M):
#     for k in range(K_L):
#         mmp = ms[mp]
#         Y_Rd_key[k, mp] = Y_Rd_key[k, mp] * exp(2j * pi * amb_num[2] * mmp * (carrier_frequency / (carrier_frequency + FL[k])))
#         Y_Rd_key[k, mp] = Y_Rd_key[k, mp] * exp(-2j * pi * amb_num[2] * mmp * (FL[k] / carrier_frequency))

y_temp_key = ifft(ifftshift( Y_Rd_key, axes=0), K_L, axis=0)
y_rd_key = y_temp_key[0:L-1, :]
Y_rD_key = fftshift(fft(y_rd_key, K_M, axis=-1), axes=-1)

plt.figure()
plt.imshow(abs(y_rd_key), aspect='auto', cmap='gray_r')
plt.grid()
plt.xlabel('pulse number')
plt.ylabel('range bin')
plt.title('Raw Fast-time/Slow-time Data pattern')

plt.figure()
mmplot(Y_rD_key)
plt.xlabel('normalized Doppler')
plt.ylabel('range bin')
plt.title('Raw Range-Doppler Matrix')

# %%
fig = plt.figure()
fig.add_subplot(121)
plt.imshow(abs(Y_Rd_key))
plt.colorbar()
plt.xlabel('slow time')
plt.ylabel('fast time frequency')
plt.title('Magnitude')
fig.add_subplot(122)
plt.imshow(angle(Y_Rd_key))
plt.colorbar()
plt.xlabel('slow time')
plt.ylabel('fast time frequency')
plt.title('Unwrapped phase')



