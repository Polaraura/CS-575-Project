import numpy as np

from numpy.linalg import norm as np_norm

from preconditioning import iter_precondition_gmres


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

    iter_precondition_time, w = iter_precondition_gmres(A, M, V, k, precondition)

    # calculate first k elements of the kth Hessenberg column
    for i in range(k + 1):
        h_k[i] = w @ V[:, i]
        w = w - h_k[i] * V[:, i]

    # h_k[k + 1] = sp_norm(w)
    h_k[k + 1] = np_norm(w)

    # check if 0
    if h_k[k + 1] == 0:
        # None for v to check in gmres (early termination with EXACT SOLUTION)
        return h_k, None, 0
    else:
        # find the new orthogonal vector in the basis of the Krylov subspace
        # assert h_k[k + 1] != 0
        v_new = w / h_k[k + 1]

    # v_new = w / h_k[k + 1]

    return h_k, v_new, iter_precondition_time
