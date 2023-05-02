import numpy as np
import scipy as sp

from numpy.linalg import norm as np_norm, \
    solve as np_solve
from scipy.sparse.linalg import norm as sp_norm, spsolve
from scipy.sparse import diags as sp_diags

from gmres import gmres
from precondition import precondition_enum


if __name__ == '__main__':
    # m_array = [600, 1300, 2500, 5000, 1000, 2000]
    # m_array = [600, 1200, 2400, 4800, 9600, 19200]
    m_array = [400, 800, 1600, 3200, 6400, 12800]
    # b_array = [10, 50, 100, 500, 1000, 5000, 10000]
    b_array = [1]

    # starting with c = 0.75, convergence is consistent
    c_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

    # TODO: do
    precondition_array = [precondition_enum.GAUSS_SEIDEL]

    for m_value in m_array:
        for b in b_array:
            print(f"m: {m_value}")
            print(f"b: {b}")

            # FIXME
            for c in c_array:
                print(f"c: {c}")

                for precondition in precondition_array:
                    print(f"precondition: {precondition}")

                    # m = 20000

                    ones_diag = np.ones((m_value - 1,))
                    twos_diag = np.ones((m_value, ))

                    # b = 900
                    a = 1
                    R = b / a
                    h = 1 / (m_value + 1)

                    # FIXME
                    # c = R * h


                    A = sp_diags([-(ones_diag - c), twos_diag + c, -ones_diag],
                                 offsets=[-1, 0, 1],
                                 format="csr")
                    # b = np.ones((m, ))
                    y = np.zeros((m_value,))
                    y[-1] = 1
                    # b = np.random.random((m, ))

                    x_true = spsolve(A, y)
                    x_gmres, error_list, k_final = \
                        gmres(A, y, precondition=precondition)

                    # print(f"x true: {x_true}")
                    # print(f"x GMRES: {x_gmres}")

                    # print(f"error: {error_list}")
                    print(f"num iterations: {k_final}")
                    print(f"norm error: {np_norm(x_true - x_gmres)}")


