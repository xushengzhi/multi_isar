# -*- coding: utf-8 -*-
'''
Creat on 2019-07-09

Authors: shengzhixu

Email: sz.xu@hotmail.com


make sure this script is running after multiisar.py using ipython console.

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.dual import eigvalsh
from scipy.signal import argrelextrema
from multi_isar.detect_local_minima import detect_local_minima
from skimage.restoration import denoise_tv_chambolle
from pylab import *

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
indx = np.delete(indx, to_delete_list)
indy = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx], ascan[indy]))
ax[1, 0].scatter(indx, indy, s=20, c=scatter_c, marker=MARKER)


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
indx = np.delete(indx, to_delete_list)
indy = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx], ascan[indy]))
ax[1, 1].scatter(indx, indy, s=20, c=scatter_c, marker=MARKER)




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
indx = np.delete(indx, to_delete_list)
indy = np.delete(indy, to_delete_list)
print('vr estimation: {}       ar estimation: {} '.format(vspan[indx], ascan[indy]))
ax[1, 2].scatter(indx, indy, s=20, c=scatter_c, marker=MARKER)

plt.tight_layout()

if save_fig:
    plt.savefig("denoising_estimation.png", dpi=400)

#%% estimation of parameters

v_est = vspan[indx]
a_est = ascan[indy]

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
plt.clim(vmin=np.max(data_fft2_db)-20, vmax=np.max(data_fft2_db))
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
dataf = Keystone(dataf, fd, fdr);
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
plt.clim(vmin=np.max(data_fft2_db)-20, vmax=np.max(data_fft2_db))
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
theta_ = theta

indY, indX = np.nonzero(car)
image_y, image_x = car.shape
y = np.linspace(-0.5, 0.5, image_y, endpoint=False)[indY]

cof1_y = c*R[i]/fc/T/2/vre/tan(deg2rad(theta_[i])) * y
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

cof1_y = c*R[i]/fc/T/2/vre/tan(deg2rad(theta_[i])) * y
cof1_x = (indX - image_x/2) *resolution/zoom

# plt.figure()
# plt.scatter(cof1_y, cof1_x)

nX, nY = rotate_target(cof1_x, cof1_y, -theta_[i])

plt.figure()
plt.scatter(nY, nX, s=MARKER_SIZE, marker=MARKER, c=MARKER_COLOR)
plt.xlim(-2, 2)
plt.ylim(-1, 5)

