import numpy as np
import scipy as sp

from numpy.linalg import norm as np_norm
from scipy.sparse.linalg import norm as sp_norm

from math import sqrt, pow


def givens_rotation(H, c_array, s_array, k):
    # get kth column of Hessenberg (nonzeros)
    h_k = H[:(k + 2), k]

    # update first k - 1 elements of h_k (except for first iteration)
    for i in range(k):
        temp = c_array[i] * h_k[i] + s_array[i] * h_k[i + 1]
        h_k[i + 1] = -s_array[i] * h_k[i] + c_array[i] * h_k[i + 1]
        h_k[i] = temp

    s_value, c_value = get_givens_rotation_values(h_k, k)

    # eliminate k, k-1 element of H
    h_k[k] = c_value * h_k[k] + s_value * h_k[k + 1]
    h_k[k + 1] = 0.0

    return h_k, s_value, c_value


def get_givens_rotation_values(h_k, k):
    h_up = h_k[k]
    h_down = h_k[k + 1]

    denominator = sqrt(pow(h_up, 2) + pow(h_down, 2))

    s_value = h_down / denominator
    c_value = h_up / denominator

    return s_value, c_value




