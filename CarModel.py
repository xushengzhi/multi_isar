# -*- coding: utf-8 -*-
'''
Creat on 2019-07-02

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



path = "multi_isar/"
file = "car.jpg"
img = plt.imread(file)
img2 = np.sum(img[750::, ...], -1)
binImg = img2 < 650

plt.figure()
plt.imshow(binImg, cmap='hot_r')

dot = np.zeros_like(binImg)
dot[::12, ::12] = 1
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

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(binImg, cmap='hot_r')
plt.scatter(Xc[::2], Yc[::2], s=10*abs(alpha), edgecolors='r')
ax.patch.set_visible(False)            # remove the frame
for spi in plt.gca().spines.values():  # remove the frame
    spi.set_visible(False)
plt.tight_layout()
plt.axis('off')




# np.savez('car.npz', Xc, Yc)


# Xc = []
# Yc = []
# x1 = [5,     4.5,     4,     3.5,     3,     2.5,      2,      1.5,      1,      0.5,     0]
# y1 = [11,   11.6,  12.2,    12.7,    13,    13.3,     13.4,    13.43,   13.48,   13.5,     13.5]
#
# x2 = [5, 5.3, 5.2, 5.1, 5.4]
# y2 = [11, 10.1, 9.2, 8.3, 7.4]
#
#
# Xc.append(x1)
# Yc.append(y1)
# plt.figure()
#
#
# plt.scatter(Xc, Yc)
# plt.ylim([-15, 15])
# plt.xlim([-15, 15])




