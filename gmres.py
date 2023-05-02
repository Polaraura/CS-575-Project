import numpy as np
import scipy as sp
import scipy.sparse

from numpy.linalg import norm as np_norm, \
    solve as np_solve
from scipy.sparse.linalg import norm as sp_norm, spsolve
from scipy.sparse import diags as sp_diags, \
    tril as sp_tril, eye as sp_eye, triu as sp_triu

from arnoldi import arnoldi
from givens_rotation import givens_rotation
from precondition import precondition_enum


# threshold=1e-10
def gmres(A, b, num_max_iter=100, threshold=1e-14, precondition=False):
    m, _ = A.shape

    # use zero vector as initial guess for x
    x = np.zeros((m, ))

    # find initial residual
    r_original = b - A @ x

    match precondition:
        case True:
            print("True")
        case False:
            print("False")
        case other:
            print("...")

    # if precondition == precondition_enum.JACOBI:
    #     M = sp_diags(A.diagonal(), offsets=0, format="csr")
    #     r = spsolve(M, r_original)
    # elif precondition

    # TODO: preconditioning
    if precondition:
        # M = sp_diags(A.diagonal(), offsets=0, format="csr")
        # r = spsolve(M, r_original)

        # M = sp_tril(A, format="csr")
        # r = spsolve(M, r_original)

        D_vector = A.diagonal()
        D_inv_vector = 1 / D_vector
        D_inv = sp_diags(D_inv_vector, offsets=0, format="csr")

        L = sp_eye(m) + sp_tril(A, k=-1, format="csr") @ D_inv
        U = sp_triu(A, format="csr")
        M = [L, U]

        z = spsolve(L, r_original)
        r = spsolve(U, z)

        # print(f"z: {z}")
        # print(f"r: {r}")
    else:
        M = None
        r = r_original

    # find initial norms
    # b_norm = sp_norm(b)
    # r_norm = sp_norm(r)
    b_norm = np_norm(b)
    r_norm = np_norm(r)
    curr_error = r_norm / b_norm

    # initialize rotation vectors s and c
    s_array = np.zeros((num_max_iter, ))
    c_array = np.zeros((num_max_iter, ))

    # initialize e1 canonical vector
    e1 = np.zeros((num_max_iter + 1, ))
    e1[0] = 1

    # save list of errors at each iteration
    error_list = [curr_error]

    # initialize the V basis of the Krylov subspace (concatenate as iteration
    # continues, may terminate early)
    V = np.zeros((m, 1))
    V[:, 0] = r / r_norm

    # Hessenberg matrix
    H = np.zeros((m + 1, 1))

    # beta * e_1
    beta_e_1 = r_norm * e1

    # store values on RHS for H
    # gamma_vector = np.zeros((m + 1, ))
    # gamma_vector[0] = r_norm
    gamma_vector = r_norm * e1

    k_final = 0
    early_stop = False

    for k in range(num_max_iter):
        # Arnoldi iteration
        V = np.concatenate((V, np.zeros((m, 1))), axis=1)
        H = np.concatenate((H, np.zeros((m + 1, 1))), axis=1)
        # print(f"V shape: {V.shape}")
        # print(f"H shape: {H.shape}")

        # TODO: preconditioning
        # FIXME: 0 error
        # print(f"k = {k}")
        H[:(k + 2), k], v_new = arnoldi(A, V, k,
                                              precondition=precondition,
                                              M=M)

        if v_new is None:
            break
        else:
            V[:, k + 1] = v_new

        # apply Givens rotation and eliminate last element in kth column of H
        H[:(k + 2), k], s_array[k], c_array[k] = \
            givens_rotation(H, c_array, s_array, k)

        # update residuals and error
        gamma_vector[k + 1] = -s_array[k] * gamma_vector[k]
        gamma_vector[k] = c_array[k] * gamma_vector[k]
        curr_error = abs(gamma_vector[k + 1]) / b_norm

        # add new error at current iteration
        error_list.append(curr_error)

        # print(f"current error: {curr_error}")

        # update stop iteration
        k_final = k

        # print(f"k: {k}")
        # print(f"early stop: {early_stop}")
        # print(f"condition: {curr_error < threshold}")

        if early_stop or curr_error < threshold:
            break

    # find the result
    # print(H[:(k_final + 1)])
    y = np_solve(H[:(k_final + 1), :(k_final + 1)],
                 gamma_vector[:(k_final + 1)])
    x = x + V[:, :(k_final + 1)] @ y

    return x, error_list, k_final + 1











