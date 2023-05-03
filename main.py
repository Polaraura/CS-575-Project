import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from time import time

from numpy.linalg import norm as np_norm, \
    solve as np_solve
from scipy.sparse.linalg import norm as sp_norm, spsolve, splu
from scipy.sparse import diags as sp_diags
# from scipy.linalg import lu_factor, lu_solve

from gmres import gmres
from utilities_enum import PreconditionEnum, MethodsEnum


def plot_figures(m_array,
                 c_array,
                 methods_array,
                 precondition_array,
                 num_iter_dict,
                 errors_dict,
                 total_runtime_dict_true,
                 total_runtime_dict_LU,
                 total_runtime_dict,
                 precondition_runtime_dict,
                 non_precondition_runtime_dict,
                 num_max_iter,
                 plot_runtime=False, plot_error=False):
    # plot figures
    # Set position of bar on X axis
    m_len = len(m_array)
    c_len = len(c_array)

    # FIXME: need to add 1 for LU (no preconditioners) and 1 for General solver
    # need to subtract one because of GMRES (other methods besides GMRES)
    methods_len = len(methods_array) - 1
    precondition_len = len(precondition_array)

    num_groups = m_len

    num_bars_per_group = precondition_len + methods_len
    barWidth = 1 / (num_bars_per_group + 1)

    br1 = np.arange(num_groups)

    bars_array = [br1]
    # already have the first bar created
    for i in range(num_bars_per_group - 1):
        prev_bars = bars_array[i]
        curr_bars = [x + barWidth for x in prev_bars]
        bars_array.append(curr_bars)

    # br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]

    #######################################
    # Plot runtime
    #######################################

    if plot_runtime:
        for c_value in c_array:
            # plt.figure(1)
            fig, ax = plt.subplots()

            # First, let's remove the top, right and left spines (figure borders)
            # which really aren't necessary for a bar chart.
            # Also, make the bottom spine gray instead of black.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')

            # Second, remove the ticks as well.
            # ax.get_minor_ticks().set_visible(False)
            # ax.tick_params(axis="both", bottom=False, left=False)
            plt.tick_params(axis="both", which="both", bottom=False, left=False)

            # # Third, add a horizontal grid (but keep the vertical grid hidden).
            # # Color the lines a light gray as well.
            # ax.set_axisbelow(True)
            # ax.yaxis.grid(True, which="both", color='#EEEEEE', linestyle='dotted')
            # ax.xaxis.grid(False)

            # plt.grid(axis='y', which='both', linestyle='dotted', linewidth=0.5)
            # plt.grid(axis='y', which='both', linestyle='dotted')

            # fig.tight_layout()

            # zorder before plotting...
            # ax.set_axisbelow(True)
            # plt.grid(True, axis="y", which="both", linestyle='dotted', zorder=0)
            # plt.grid(True, axis="y", color='#EEEEEE', which="both",
            #          linestyle='dotted')
            # check if the grid looks good...

            # ax = fig.add_axes([0, 0, 1, 1])

            # c_value = c_array[0]
            bottom_bar_array = []

            # print(non_precondition_runtime_dict)

            # for m_index, m_value in enumerate(m_array):
            for method_index, method_type in enumerate(methods_array):
                match method_type:
                    case MethodsEnum.GENERAL_SOLVER:
                        total_runtime_array_general_c = [total_runtime_dict_true[m_value][c_value]
                                                         for m_value in m_array]

                        # FIXME: always set to 0
                        plt.bar(bars_array[0],
                                total_runtime_array_general_c,
                                width=barWidth,
                                label=f"General Solver (built-in)",
                                log=1)

                    case MethodsEnum.LU:
                        total_runtime_array_LU_c = [total_runtime_dict_LU[m_value][c_value]
                                                         for m_value in m_array]

                        # FIXME: always set to 1
                        plt.bar(bars_array[1],
                                total_runtime_array_LU_c,
                                width=barWidth,
                                label=f"LU (built-in)",
                                log=1)

                    case MethodsEnum.GMRES:
                        gmres_offset_index = methods_len

                        for precondition_index, precondition_type in enumerate(precondition_array):
                            precondition_index_adjusted = precondition_index + gmres_offset_index

                            # print(f"precondition: {precondition_type}")
                            if precondition_type is None:
                                total_runtime_array_c = [total_runtime_dict[m_value][c_value][precondition_type]
                                                         for m_value in m_array]

                                # color='...'
                                # blue - b
                                plt.bar(bars_array[precondition_index_adjusted],
                                        total_runtime_array_c,
                                        width=barWidth,
                                        label=f"Not Preconditioned",
                                        log=1)
                            else:
                                precondition_name = str.lower(precondition_type.name).capitalize()

                                non_precondition_runtime_array_c = [non_precondition_runtime_dict[m_value][c_value][precondition_type]
                                                                    for m_value in m_array]
                                bottom_bar_array.append(non_precondition_runtime_array_c)

                                # print(non_precondition_runtime_array_c)
                                # print(non_precondition_runtime_dict)

                                # changed from br3 -- stack the repartition time with new hybrid time
                                # change so output[3] is on bottom -- new repartitioned hybrid time
                                # red - r
                                plt.bar(bars_array[precondition_index_adjusted],
                                        non_precondition_runtime_array_c,
                                        width=barWidth,
                                        label=f"Un-precondition {precondition_name}",
                                        log=1)

                                precondition_runtime_array_c = [precondition_runtime_dict[m_value][c_value][precondition_type]
                                                                for m_value in m_array]

                                # bottom bar EXCLUDE None...
                                # green - g
                                plt.bar(bars_array[precondition_index_adjusted],
                                        precondition_runtime_array_c,
                                        width=barWidth,
                                        label=f"Precondition {precondition_name}",
                                        bottom=bottom_bar_array[precondition_index - 1],
                                        log=1)

                    case _:
                        raise ValueError("Invalid method type...")

            ##########################
            # plot details
            plt.xlabel(f"Array Size")
            plt.ylabel(f"Time (s)")
            plt.title(f"Runtime (c={c_value})")

            offset = ((1 / 2) * (num_bars_per_group - 1)) * barWidth

            plt.legend(loc='upper left')

            plt.xticks([r + offset for r in range(m_len)],
                       m_array)

            plt.grid(True, axis='y', which='both', linestyle='dotted')

            plt.savefig(f"new_figures/gmres_runtime_timings_plot_c={c_value}.png", bbox_inches="tight")

        # print(bottom_bar_array)

    #######################################
    # Plot error
    #######################################

    if plot_error:
        for c_value in c_array:
            for m_value in m_array:
                fig, ax = plt.subplots()

                for precondition_index, precondition_type in enumerate(precondition_array):
                    # errors_array_c = [errors_dict[m_value][c_value][precondition_type]
                    #                                     for m_value in m_array]

                    match precondition_type:
                        case None:
                            method_name = "GMRES"
                        case other:
                            method_name = "GMRES "
                            method_name += str.lower(other.name).capitalize()

                    # if precondition_type is None:
                    #     method_name = "GMRES"
                    # else:
                    #     method_name = str.lower(precondition_type.name).capitalize()

                    num_iter = num_iter_dict[m_value][c_value][precondition_type]
                    errors_array_c_m = errors_dict[m_value][c_value][precondition_type]

                    # FIXME: just use the length of the residual (errors array) -- iteration count differs if it
                    #  stopped early
                    num_iter_residual = len(errors_array_c_m)
                    # plt.semilogy(list(range(num_iter + 1)), errors_array_c_m,
                    #              '.--',
                    #              label=method_name)
                    plt.semilogy(list(range(num_iter_residual)), errors_array_c_m,
                                 '.--',
                                 label=method_name)

                # plot details
                plt.xlabel(f"Iteration")
                plt.ylabel(r"Residual $\min{||\beta e_1 - \bar{H}_m y||}_2$")
                plt.title(f"Residual (c={c_value}, m={m_value})")

                plt.legend(loc='upper right')

                # FIXME: doesn't work...to add minor ticks in
                ax.set_yscale("log")
                plt.tick_params(axis="both", which="both", left=True, bottom=True)
                locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

                plt.grid(True, which='both', linestyle='dotted')

                plt.savefig(f"new_figures/gmres_residual_plot_c={c_value}_m={m_value}.png", bbox_inches="tight")


if __name__ == '__main__':
    num_max_iter = 100
    threshold = 1e-14

    # m_array = [600, 1300, 2500, 5000, 1000, 2000]
    # m_array = [600, 1200, 2400, 4800, 9600, 19200]
    m_array = [400, 800, 1600, 3200, 6400, 12800]
    # m_array = [6400, 12800]

    # a = []
    # b = []
    # for i in range(3):
    #     b = [1, 2, 3]
    #     b.append(i)
    #     a.append(b)
    # print(a)

    # b_array = [10, 50, 100, 500, 1000, 5000, 10000]
    # b_array = [1]

    # starting with c = 0.75, convergence is consistent
    c_array = [0.5, 1, 10, 100, 1000]
    # c_array = [0.5]

    # TODO: do
    # precondition_array = [precondition_enum.JACOBI,
    #                       precondition_enum.GAUSS_SEIDEL,
    #                       precondition_enum.SYMMETRIC_GAUSS_SEIDEL]
    # precondition_array = [precondition_enum.name
    #                       for p_enum in precondition_enum]
    methods_array = [m_enum for m_enum in MethodsEnum]
    precondition_array = [None] + [p_enum for p_enum in PreconditionEnum]

    num_iter_dict = {}
    errors_dict = {}
    total_runtime_dict_true = {}
    total_runtime_dict_LU = {}
    total_runtime_dict = {}
    non_precondition_runtime_dict = {}
    precondition_runtime_dict = {}

    print(f"len enum: {len(PreconditionEnum)}")
    print()

    for m_value in m_array:
        num_iter_dict[m_value] = {}
        errors_dict[m_value] = {}
        total_runtime_dict_true[m_value] = {}
        total_runtime_dict_LU[m_value] = {}
        total_runtime_dict[m_value] = {}
        non_precondition_runtime_dict[m_value] = {}
        precondition_runtime_dict[m_value] = {}

        for c_value in c_array:
            num_iter_dict[m_value][c_value] = {}
            errors_dict[m_value][c_value] = {}
            total_runtime_dict_true[m_value][c_value] = {}
            total_runtime_dict_LU[m_value][c_value] = {}
            total_runtime_dict[m_value][c_value] = {}
            non_precondition_runtime_dict[m_value][c_value] = {}
            precondition_runtime_dict[m_value][c_value] = {}

            print(f"m: {m_value}")
            print(f"c: {c_value}")
            print()

            #########################################

            # m = 20000

            ones_diag = np.ones((m_value - 1,))
            twos_diag = np.ones((m_value,))

            # b = 900
            # a = 1
            # R = b / a
            # h = 1 / (m_value + 1)

            # FIXME
            # c = R * h

            A = sp_diags(
                [-(ones_diag - c_value), twos_diag + c_value, -ones_diag],
                offsets=[-1, 0, 1],
                format="csr")
            # b = np.ones((m, ))
            y = np.zeros((m_value,))
            y[-1] = 1
            # b = np.random.random((m, ))

            start_time_true = time()
            x_true = spsolve(A, y)
            end_time_true = time()

            start_time_LU = time()
            # lu, piv = lu_factor(A)
            # x_LU = lu_solve((lu, piv), y)

            # find LU factorization and solve it (get Solve object and call solve on it)
            LU_solve = splu(A)
            x_LU = LU_solve.solve(y)

            end_time_LU = time()

            print(f"LU error: {np_norm(x_true - x_LU)}")
            print()

            total_time_true = end_time_true - start_time_true
            total_time_LU = end_time_LU - start_time_LU

            # FIXME: not dependent on precondition
            total_runtime_dict_true[m_value][c_value] = \
                total_time_true
            total_runtime_dict_LU[m_value][c_value] = \
                total_time_LU

            #########################################

            for precondition in precondition_array:
                print(f"precondition: {precondition}")

                start_time = time()
                x_gmres, error_list, k_final, precondition_time = \
                    gmres(A, y, num_max_iter=num_max_iter,
                          threshold=threshold,
                          precondition=precondition)
                end_time = time()

                total_time = end_time - start_time

                num_iter_dict[m_value][c_value][precondition] = k_final
                errors_dict[m_value][c_value][precondition] = error_list
                total_runtime_dict[m_value][c_value][precondition] = \
                    total_time
                precondition_runtime_dict[m_value][c_value][precondition] = \
                    precondition_time

                non_precondition_time = total_time - precondition_time
                non_precondition_runtime_dict[m_value][c_value][precondition] = \
                    non_precondition_time

                # print(f"x true: {x_true}")
                # print(f"x GMRES: {x_gmres}")

                # print(f"error: {error_list}")
                print(f"num iterations: {k_final}")
                print(f"norm error: {np_norm(x_true - x_gmres)}")
                print(f"errors list: {error_list}")
                print(f"total runtime (true): {total_time_true}")
                print(f"total runtime: {total_time}")
                print(f"precondition time: {precondition_time}")
                print(f"non-precondition time: {non_precondition_time}")
                print()

    # plot the graphs
    plot_figures(m_array, c_array, methods_array, precondition_array, num_iter_dict, errors_dict,
                 total_runtime_dict_true, total_runtime_dict_LU, total_runtime_dict, precondition_runtime_dict,
                 non_precondition_runtime_dict, num_max_iter, plot_runtime=True, plot_error=True)
