import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt

def power_iteration(A, u, max_iter=100):
    '''
    Simple power iteration for largest eigenvalue
    '''
    for i in range(max_iter):
        w = A.dot(u)
        u = w/np.linalg.norm(w)
        e = u.T.conj().dot(A).dot(u)

    return e


def inverse_iteration(A, u, max_iter=100):
    pass


def rayleigh_quotient_iteration(A, u, max_iter=100):
    pass


def qr_iteration(A):
    pass


def simutaneous_iteration(A, Q, max_iter=1):
    '''
    simultaneous iteration is power iteration applied to several vectors
    '''
    for i in range(max_iter):
        Z = A.dot(Q)
        Q, R = qr(Z)	# reduce QR factorization of Z

    return R


if __name__ == "__main__":
    import time

    len_size = 256
    X, Y = np.meshgrid(np.arange(len_size), np.arange(len_size))
    Z = np.exp(2j*np.pi*(0.2*X-0.2*Y + 0.01*X*X))

    plt.imshow(np.log10(abs(np.fft.fft2(Z))))
    # Z = Z.dot(Z.T.conj())
    # tic = time.time()
    # np.linalg.eigvalsh(Z)
    # print(time.time() - tic)
    #
    #
    # I = np.identity(len_size)
    # tic = time.time()
    # simutaneous_iteration(Z, I, max_iter=5)
    # print(time.time() - tic)
			
