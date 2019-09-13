# -*- coding: utf-8 -*-
'''
Creat on 2019-04-23

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import log10


def normalize(data, mode='01'):
    if mode == '01':
        return (data - np.min(data))/(np.max(data) - np.min(data))
    elif mode == '80':
        return (data - np.max(data))


def awgn(sig, snrdb, sigpower=0):
    """
    Additive white gaussian noise.  Assumes signal power is 0 dBW
    """
    L = sig.size
    sigpower = 10*np.log10(np.sum(np.abs(sig)**2)/L)

    if sp.iscomplexobj(sig):
        noise = (sp.randn(*sig.shape) + 1j*sp.randn(*sig.shape))/np.sqrt(2)
    else:
        noise = sp.randn(*sig.shape)
    noisev = 10**((sigpower - snrdb)/20)
    return sig + noise*noisev


def cart2pol(x, y):
    '''
    Cartesian to polar axis
    :param x:
    :param y:
    :return: float radius and float angle in radians
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    '''
    Polar to Cartesian axis
    :param rho:
    :param phi:
    :return: float x and float y axis
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def db(mag):
    '''
    return power dB
    :param mag:
    :return: float
    '''
    return 20*np.log10(abs(mag))


def mplot(pic, clim=40, *args, **kwargs):
    '''
    Radar map with clim
    :param pic:
    :param clim:
    :param args:
    :param kwargs:
    :return: None
    '''
    vmax = np.max(pic)
    plt.imshow(pic, aspect='auto', cmap='gray_r', *args, **kwargs)
    plt.colorbar()
    plt.clim([vmax-clim, vmax])


def entropy(vector):
    info = vector/np.sum(vector)
    return -np.sum(info * log10(abs(info)))


def renyi(vector, alpha=50):
    if alpha == 1:
        return entropy(vector)
    else:
        # TODO: Gaussian Kernel
        info = vector/np.sum(vector)
        return 1/(1-alpha) * log10(np.sum(abs(info)**alpha))


def tsallis(vector, q=3.5):
    if q == 1:
        return entropy(vector)
    else:
        # TODO: Kernel
        info = vector / np.sum(vector)
        return 1/(q-1) - np.sum(info**q/(q-1))

def mmplot(mat, clim=40):
    '''
    power dB plot with clim and normalization to 0dB
    :param mat: input matrix
    :param clim: axis clim
    :return: None, plt object
    '''
    mat = 20 * log10(abs(mat))
    mat = mat - np.max(mat)
    plt.imshow(mat, cmap='jet', aspect='auto')
    plt.clim([-clim, 0])
    plt.colorbar()

# image contrast method
def spatial_mean(image, *args, **kwargs):
    energy = np.sum(abs(image))
    return energy/image.size

def image_constrast(image, *args, **kwargs):

    spatial_mean_image = spatial_mean(image, *args, **kwargs)
    return np.sqrt(spatial_mean((image - spatial_mean_image)**2)) / spatial_mean_image



if __name__ == "__main__":
    pass