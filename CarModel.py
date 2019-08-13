# -*- coding: utf-8 -*-
'''
Creat on 2019-07-02

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



path = "data/"
file = "car.jpg"
img = plt.imread(path + file)
img2 = np.sum(img[750::, ...], -1)
binImg = img2 < 650

plt.figure()
plt.imshow(binImg, cmap='hot_r')

dot = np.zeros_like(binImg)
dot[::12, ::15] = 1
dotImg = dot * binImg

plt.figure()
plt.imshow(dotImg)


x, y = binImg.shape
noise = np.random.rand(x*y).reshape(x, y) > 0
noiseImg = binImg & noise
plt.figure()
plt.imshow(binImg & noise, cmap='hot_r')


np.sum(noiseImg)
Yc, Xc = np.nonzero(dotImg)
# Xc.shape

save_fig = False
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(binImg, cmap='hot_r')
plt.scatter(Xc[::2], Yc[::2], s=18, edgecolors='r')
ax.patch.set_visible(False)            # remove the frame
for spi in plt.gca().spines.values():  # remove the frame
    spi.set_visible(False)
plt.tight_layout()
plt.axis('off')
if save_fig:
    plt.savefig('car_model.png', dpi=300)


def complex_amplitude(low_alpha):
    COF = np.sqrt(2)/2
    real = COF * (low_alpha + (1-low_alpha) * np.random.rand())
    imag = COF * (low_alpha + (1-low_alpha) * np.random.rand())
    return (real+ 1j * imag)

def varying_amplitude(X, low_alpha=0.5):
    k = X.size
    alpha = np.zeros_like(X, dtype=complex)
    for i in range(k):
        alpha[i] = complex_amplitude(low_alpha) * np.random.randn() * 10

    return alpha


alpha = varying_amplitude(Xc, 0.5)


# %%
fig = plt.figure(figsize=[12, 6])
ax = fig.add_subplot(1, 1, 1)
plt.imshow(binImg, cmap='hot_r')
plt.scatter(Xc[::], Yc[::], s=15*abs(alpha), edgecolors='r')
ax.patch.set_visible(False)            # remove the frame
for spi in plt.gca().spines.values():  # remove the frame
    spi.set_visible(False)
plt.tight_layout()
plt.axis('off')
# if save_fig:
#     plt.savefig("car_model.png", dpi=300)


# np.savez('car.npz', Xc, Yc, alpha)




