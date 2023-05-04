from time import time

from scipy.sparse import diags as sp_diags, tril as sp_tril, eye as sp_eye, triu as sp_triu
from scipy.sparse.linalg import spsolve

from utilities_enum import PreconditionEnum


def initial_precondition(A, m, precondition, r_original):
    # total precondition time
    total_precondition_time = 0
    precondition_start_time = time()
    match precondition:
        case PreconditionEnum.JACOBI:
            M = sp_diags(A.diagonal(), offsets=0, format="csr")
            r = spsolve(M, r_original)
        case PreconditionEnum.GAUSS_SEIDEL:
            M = sp_tril(A, format="csr")
            r = spsolve(M, r_original)
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            D_vector = A.diagonal()
            D_inv_vector = 1 / D_vector
            D_inv = sp_diags(D_inv_vector, offsets=0, format="csr")

            L = sp_eye(m) + sp_tril(A, k=-1, format="csr") @ D_inv
            U = sp_triu(A, format="csr")
            M = [L, U]

            z = spsolve(L, r_original)
            r = spsolve(U, z)
        # default case (None)
        case _:
            M = None
            r = r_original

    precondition_end_time = time()
    total_precondition_time += (precondition_end_time - precondition_start_time)

    return M, r, total_precondition_time


def iter_precondition_gmres(A, M, V, k, precondition):
    # apply the preconditioning at each iteration step
    precondition_start_time = time()

    match precondition:
        case PreconditionEnum.JACOBI | PreconditionEnum.GAUSS_SEIDEL:
            w = spsolve(M, A @ V[:, k])
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            L, U = M
            z = spsolve(L, A @ V[:, k])
            w = spsolve(U, z)
        # default case (None)
        case _:
            w = A @ V[:, k]

    precondition_end_time = time()
    iter_precondition_time = precondition_end_time - precondition_start_time
    return iter_precondition_time, w


def iter_precondition_conjugate_gradient(A, M, r, precondition):
    # apply the preconditioning at each iteration step
    precondition_start_time = time()

    match precondition:
        case PreconditionEnum.JACOBI | PreconditionEnum.GAUSS_SEIDEL:
            z = spsolve(M, A @ r)
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            L, U = M
            z = spsolve(L, A @ r)
            z = spsolve(U, z)
        # default case (None)
        case _:
            z = A @ r

    precondition_end_time = time()
    iter_precondition_time = precondition_end_time - precondition_start_time

    return iter_precondition_time, z
