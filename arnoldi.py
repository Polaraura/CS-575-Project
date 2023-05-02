import numpy as np
import scipy as sp

from time import time

from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import norm as sp_norm, spsolve
from scipy.sparse import triu as sp_triu

from precondition import precondition_enum


def arnoldi(A, V, k, precondition=False, M=None):
    """
    Computes one iteration of Arnoldi iteration given the iteration index k

    :param precondition:
    :param A:
    :param V:
    :param k:
    :return:
    """

    m, _ = A.shape

    # inialize k + 1 nonzero elements of H along column k
    # k starts at 0...
    h_k = np.zeros((k + 2, ))

    precondition_start_time = time()
    match precondition:
        case precondition_enum.JACOBI | precondition_enum.GAUSS_SEIDEL:
            w = spsolve(M, A @ V[:, k])
        case precondition_enum.SYMMETRIC_GAUSS_SEIDEL:
            L, U = M
            z = spsolve(L, A @ V[:, k])
            w = spsolve(U, z)
        # default case (None)
        case _:
            w = A @ V[:, k]
    precondition_end_time = time()
    iter_precondition_time = precondition_end_time - precondition_start_time

    # if precondition:
    #     # w = spsolve(M, A @ V[:, k])
    #
    #     L, U = M
    #     z = spsolve(L, A @ V[:, k])
    #     w = spsolve(U, z)
    # else:
    #     w = A @ V[:, k]

    # calculate first k elements of the kth Hessenberg column
    for i in range(k + 1):
        h_k[i] = w @ V[:, i]
        w = w - h_k[i] * V[:, i]

    # h_k[k + 1] = sp_norm(w)
    h_k[k + 1] = np_norm(w)

    # check if 0
    if h_k[k + 1] == 0:
        return h_k, None, 0
    else:
        # assert h_k[k + 1] != 0
        v_new = w / h_k[k + 1]

    # v_new = w / h_k[k + 1]

    return h_k, v_new, iter_precondition_time





