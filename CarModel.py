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

dot = np.zeros_like(binImg)
dot[::10, ::10] = 1
binImg = dot * binImg

plt.figure()
plt.imshow(binImg)


x, y = binImg.shape
noise = np.random.rand(x*y).reshape(x, y) > 0
dotImg = binImg & noise
plt.figure()
plt.imshow(binImg & noise, cmap='hot_r')


np.sum(dotImg)
Yc, Xc = np.nonzero(dotImg)
Xc.shape
plt.figure()
plt.scatter(Xc, Yc)



np.savez('car.npz', Xc, Yc)


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




