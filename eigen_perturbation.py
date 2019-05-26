# -*- coding: utf-8 -*-
'''
Creat on 2019-05-15

Authors: shengzhixu

Email: 

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import *



'''

Slow and Not stable !!!!!!!!!!!!!!!!!!!!!!!!!

'''


def perturbation(evalue, evector, dA):

    # if not np.all(A == A.conj().T):
    #     raise ValueError

    # if not A.dot(evector) == evalue*evector:
    #     raise ValueError

    L = A.shape[0]

    new_evalue = np.zeros_like(evalue)
    new_vector = np.zeros_like(A)

    for i in range(L):
        veci = evector[i, :].reshape(1, L)
        new_evalue[i] = evalue[i] + (veci.conj().dot(dA).dot(veci.reshape(L, 1)))[0, 0]

        for j in range(L):
            vecj = evector[j, :].reshape(1, L)
            if i != j:
                new_vector[i, :] += (vecj.conj().dot(dA).dot(veci.reshape(L, 1)))[0, 0]*vecj[0, :]\
                                   /(evalue[i]- evalue[j])
        new_vector[i, :] += veci[0, :]

    return new_evalue, new_vector

len = 20
A = np.random.randn(len**2).reshape(len, len) + 1j * np.random.randn(len**2).reshape(len, len)
B = A.dot(A.T.conj())

U, V = np.linalg.eig(B) # eigenvalues and eigenvectors


X, Y = np.meshgrid(np.arange(len), np.arange(len))

C = A*exp(2j*pi*0.0001*X*Y)
D = C.dot(C.T.conj())
dB = D - B

P, Q = perturbation(U, V, dB)

M, L = np.linalg.eig(D)










