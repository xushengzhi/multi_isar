# -*- coding: utf-8 -*-
'''
Creat on 2019-04-10

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt





if __name__ == '__main__':

    N = 32
    A = np.random.randn(N**2).reshape(N)
    S = A.dot(A.T)
