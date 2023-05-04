from math import sqrt
from time import time

import numpy as np

from utilities_enum import PreconditionEnum
from preconditioning import initial_precondition, \
    iter_precondition_conjugate_gradient


def conjugate_gradient(A, y, num_max_iter=100, threshold=1e-14, precondition=None):
    """
    Conjugate gradient

    Referenced from https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB_/_GNU_Octave

    Precondition add-on referenced in Algorithm 9.1 of Saad (algorithm structure modeled after this)
    """

    m, _ = A.shape

    # use zero vector as initial guess for x
    x = np.zeros((m,))

    # find initial residual and polynomial p
    r_original = y - A @ x
    r = r_original

    # apply initial preconditioning
    M, z, total_precondition_time = \
        initial_precondition(A, m, precondition, r_original)

    # set p to the solution of the preconditioned system
    p = z

    # keep track of residuals
    residual_list = []

    # 2-norm squared of residual r
    residual_old = r @ z
    residual_list.append(residual_old)

    # iteration when CG finishes
    k_final = 0

    b_len = len(y)

    # max number of iterations bounded by m (size of b)
    # FIXME: took too long...restricted with num_max_iter instead
    for i in range(num_max_iter):
        # save product of Ap as it will be used multiple times
        Ap = A @ p

        # constant alpha
        alpha = residual_old / (p @ Ap)

        # update solution and new residual (with norm)
        x = x + alpha * p
        r = r - alpha * Ap

        # apply preconditioning step
        iter_precondition_time, z = iter_precondition_conjugate_gradient(A, M, r, precondition)

        # update precondition time
        total_precondition_time += iter_precondition_time

        # update residuals BEFORE getting p (calculating beta with new/old residuals)
        residual_new = r @ z
        residual_list.append(residual_new)

        # if not, continue with update on p and update old residual
        beta = residual_new / residual_old

        p = r + beta * p
        residual_old = residual_new

        # update iteration number
        k_final = i

        # break if threshold of new norm of residual is met
        if sqrt(residual_new) < threshold:
            break

    return x, residual_list, k_final + 1, total_precondition_time
