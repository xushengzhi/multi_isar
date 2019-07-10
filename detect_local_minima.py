# -*- coding: utf-8 -*-
'''
Creat on 2019-07-10

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = np.bitwise_xor(local_min, eroded_background)
    return np.where(detected_minima)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    denoising_method = denoise_tv_chambolle
    X, Y = np.meshgrid(np.arange(100)*0.2, np.arange(100)*0.2)
    Z = np.sin(X) * np.cos(Y) + np.random.randn(100, 100)*0.1
    arr = detect_local_minima(denoising_method(Z, weight=1, multichannel=True))

    plt.figure()
    plt.imshow(Z)
    plt.scatter(arr[1], arr[0], c='r')
    plt.show()
