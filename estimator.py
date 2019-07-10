# -*- coding: utf-8 -*-
'''
Creat on 2019-07-09

Authors: shengzhixu

Email: sz.xu@hotmail.com


make sure this script is running after multiisar.py

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from multi_isar.detect_local_minima import detect_local_minima
from skimage.restoration import denoise_tv_chambolle

plt.jet()
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharex=True, sharey=True)
weight = 0.5
weight=weight

ax[0, 0].imshow(me1, aspect='auto')
ax[0, 0].axis('off')
ax[0, 0].set_title("Me1")

ax[1, 0].imshow(denoise_tv_chambolle(me1), aspect='auto')
ax[1, 0].axis('off')
ax[1, 0].set_title("Me1 denoising")

arr = detect_local_minima(-denoise_tv_chambolle(me1, weight=weight))
ax[1, 0].scatter(arr[1], arr[0], s=20, c='w')

ax[0, 1].imshow(me2, aspect='auto')
ax[0, 1].axis('off')
ax[0, 1].set_title("Me2")

ax[1, 1].imshow(denoise_tv_chambolle(me2), aspect='auto')
ax[1, 1].axis('off')
ax[1, 1].set_title("Me2 denoising")

arr = detect_local_minima(-denoise_tv_chambolle(me2, weight=weight))
ax[1, 1].scatter(arr[1], arr[0], s=20, c='w')

ax[0, 2].imshow(me_com, aspect='auto')
ax[0, 2].axis('off')
ax[0, 2].set_title("MeCOM")

ax[1, 2].imshow(denoise_tv_chambolle(me_com), aspect='auto')
ax[1, 2].axis('off')
ax[1, 2].set_title("MeCOM denoising")

arr = detect_local_minima(-denoise_tv_chambolle(me_com, weight=weight))

ax[1, 2].scatter(arr[1], arr[0], s=20, c='w')

#%% estimation of parameters
indy, indx = arr

v_est = vspan[indx]
a_est = ascan[indy]




#%% thresholding
zoom = 2
dataf = fft2((data)* exp(2j*pi*(v_est[0]*fdr*X*Y + a_est[0]*fa*X*Y*Y + a_est[0]*frs*Y*Y)), [zoom*n, zoom*m])
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
car1 = (data_fft2_db > (max_spec-threshold))
fig.add_subplot(1, 2, 2)
car_dilation = dilation(dilation(dilation(car1)))
plt.imshow(np.fliplr(car_dilation.T),
           aspect='auto',
           extent=[0, range_domain, -max_unambiguous_velocity, max_unambiguous_velocity],
           interpolation='none',
           cmap='gray_r')
plt.xlabel('Range (m)')
plt.ylabel('Doppler (m/s)')



dataf = fft2((data)* exp(2j*pi* ( v_est[1]*fdr*X*Y + a_est[1]*fa*X*Y*Y + a_est[1]*frs*Y*Y)), [zoom*n, zoom*m])
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


max_spec = np.max(data_fft2_db)
car2 = (data_fft2_db > (max_spec-threshold))
fig.add_subplot(1, 2, 2)
car_dilation = dilation(dilation(dilation(car2)))
plt.imshow(np.fliplr(car_dilation.T),
           aspect='auto',
           extent=[0, range_domain, -max_unambiguous_velocity, max_unambiguous_velocity],
           interpolation='none',
           cmap='gray_r')
plt.xlabel('Range (m)')
plt.ylabel('Doppler (m/s)')
