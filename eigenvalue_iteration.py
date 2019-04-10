import numpy as np
from scipy.linalg import qr, norm
import matplotlib.pyplot as plt

def power_iteration(A, u, max_iter=100):
    '''
    Simple power iteration for largest eigenvalue
    '''
    for i in range(max_iter):
        w = A.dot(u)
        u = w/np.linalg.norm(w)
        e = u.T.conj().dot(A).dot(u)

    return e, u


def inverse_iteration(A, u, max_iter=100):
    pass


def rayleigh_quotient_iteration(A, u, max_iter=100):
    pass


def qr_iteration(A):
    pass


def simutaneous_iteration(A, Q=None, max_iter=1):
    '''
    simultaneous iteration is power iteration applied to several vectors
    '''

    if Q is None:
        Q = np.eye(A.shape[0])
    for i in range(max_iter):
        Z = A.dot(Q)
        Q, R = qr(Z)	# reduce QR factorization of Z

    return np.diag(R), Q


if __name__ == "__main__":
    import time

    #%% TEST SIMULTANEOUS ITERATION
    len_size = 128
    A = np.random.randn(len_size**2).reshape(len_size, len_size) + \
        1j*np.random.randn(len_size**2).reshape(len_size, len_size)
    H = A.dot(A.T.conj())
    u = np.sort(np.linalg.eigvalsh(H))[::-1]

    Q = np.identity(128)
    error = []
    for i in range(1000):
        v, Q = simutaneous_iteration(H, Q, max_iter=1)
        error.append(sum( (u - np.sort(v)[::-1])**2))

    plt.semilogy(error)

    print(np.allclose( Q.dot(np.diag(v)).dot(Q.T.conj()), H))
    # len_size = 12
    # X, Y, Z = np.meshgrid(np.arange(len_size), np.arange(len_size), np.arange(len_size))
    # D = np.exp(2j*np.pi*(0.01*X*Y))
    # Dr = np.reshape(np.transpose(D, [0, 1, 2]), [128*128, 128])
    # print(np.linalg.matrix_rank(Dr))


    # len_size = 256
    # X, Y = np.meshgrid(np.arange(len_size), np.arange(len_size))
    # Z = np.random.randn(len_size*len_size).reshape(len_size, len_size) + 1j*np.random.randn(len_size*len_size).reshape(len_size, len_size)
    # C = Z.dot(Z.T.conj())
    #
    # a = np.linalg.norm(C, 2)
    # vec = np.zeros((256, 1), dtype=complex)
    # vec[0] = 1
    # error = np.zeros((50, ))
    # for i in range(50):
    #     e, vec = power_iteration(C, vec, 1)
    #     error[i] = abs(e-a)
    # plt.plot(error)


    # plt.imshow(np.log10(abs(np.fft.fft2(Z))))
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
			
