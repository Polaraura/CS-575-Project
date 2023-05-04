import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.cm import get_cmap
from matplotlib import rc
rc('text', usetex=True)

from time import time

from numpy.linalg import norm as np_norm, \
    solve as np_solve
from scipy.sparse.linalg import norm as sp_norm, spsolve, splu
from scipy.sparse import diags as sp_diags
# from scipy.linalg import lu_factor, lu_solve

from gmres import gmres
from conjugate_gradient import conjugate_gradient
from utilities_enum import PreconditionEnum, BuiltInMethodsEnum, ImplementationMethodsEnum


def plot_figures(m_array,
                 c_array,
                 built_in_methods_array,
                 implementation_methods_array,
                 precondition_array,
                 num_iter_dict,
                 total_runtime_dict_true,
                 total_runtime_dict_LU,
                 gmres_residual_dict,
                 gmres_total_runtime_dict,
                 gmres_precondition_runtime_dict,
                 gmres_non_precondition_runtime_dict,
                 cg_residual_dict,
                 cg_total_runtime_dict,
                 cg_precondition_runtime_dict,
                 cg_non_precondition_runtime_dict,
                 num_max_iter,
                 plot_runtime=False, plot_error=False):
    # plot figures
    # Set position of bar on X axis
    m_len = len(m_array)
    c_len = len(c_array)

    # FIXME: need to add 1 for LU (no preconditioners) and 1 for General solver
    # need to subtract one because of GMRES (other methods besides GMRES)
    built_in_methods_len = len(built_in_methods_array)
    implementation_methods_len = len(implementation_methods_array)
    precondition_len = len(precondition_array)

    num_groups = m_len

    # each implementation should use the preconditioners and the built-in should NOT
    num_bars_per_group = implementation_methods_len * precondition_len + built_in_methods_len
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

            # color = cm.rainbow(np.linspace(0, 1, n))

            name = "tab20"
            cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
            colors = cmap.colors  # type: list
            ax.set_prop_cycle(color=colors)

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

            # for m_index, m_value in enumerate(m_array):
            for method_index, method_type in enumerate(built_in_methods_array):
                match method_type:
                    case BuiltInMethodsEnum.GENERAL_SOLVER:
                        total_runtime_array_general_c = [total_runtime_dict_true[m_value][c_value]
                                                         for m_value in m_array]

                        # FIXME: always set to 0
                        plt.bar(bars_array[0],
                                total_runtime_array_general_c,
                                width=barWidth,
                                label=f"General Solver (built-in)",
                                log=1)

                    case BuiltInMethodsEnum.LU:
                        total_runtime_array_LU_c = [total_runtime_dict_LU[m_value][c_value]
                                                         for m_value in m_array]

                        # FIXME: always set to 1
                        plt.bar(bars_array[1],
                                total_runtime_array_LU_c,
                                width=barWidth,
                                label=f"LU (built-in)",
                                log=1)

            for method_index, method_type in enumerate(implementation_methods_array):
                # c_value = c_array[0]
                bottom_bar_array = []

                # print(non_precondition_runtime_dict)

                match method_type:
                    case ImplementationMethodsEnum.GMRES:
                        gmres_offset_index = built_in_methods_len

                        for precondition_index, precondition_type in enumerate(precondition_array):
                            precondition_index_adjusted = precondition_index + gmres_offset_index

                            # print(f"precondition: {precondition_type}")
                            if precondition_type is None:
                                total_runtime_array_c = [gmres_total_runtime_dict[m_value][c_value][precondition_type]
                                                         for m_value in m_array]

                                # color='...'
                                # blue - b
                                plt.bar(bars_array[precondition_index_adjusted],
                                        total_runtime_array_c,
                                        width=barWidth,
                                        label=f"GMRES Not Preconditioned",
                                        log=1)
                            else:
                                precondition_name = str.lower(precondition_type.name).capitalize()

                                non_precondition_runtime_array_c = [gmres_non_precondition_runtime_dict[m_value][c_value][precondition_type]
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
                                        label=f"GMRES Un-precondition {precondition_name}",
                                        log=1)

                                precondition_runtime_array_c = [gmres_precondition_runtime_dict[m_value][c_value][precondition_type]
                                                                for m_value in m_array]

                                # bottom bar EXCLUDE None...
                                # green - g
                                plt.bar(bars_array[precondition_index_adjusted],
                                        precondition_runtime_array_c,
                                        width=barWidth,
                                        label=f"GMRES Precondition {precondition_name}",
                                        bottom=bottom_bar_array[precondition_index - 1],
                                        log=1)

                    case ImplementationMethodsEnum.CG:
                        cg_offset_index = built_in_methods_len + precondition_len

                        for precondition_index, precondition_type in enumerate(precondition_array):
                            precondition_index_adjusted = precondition_index + cg_offset_index

                            # print(f"precondition: {precondition_type}")
                            if precondition_type is None:
                                total_runtime_array_c = [cg_total_runtime_dict[m_value][c_value][precondition_type]
                                                         for m_value in m_array]

                                # color='...'
                                # blue - b
                                plt.bar(bars_array[precondition_index_adjusted],
                                        total_runtime_array_c,
                                        width=barWidth,
                                        label=f"CG Not Preconditioned",
                                        log=1)
                            else:
                                precondition_name = str.lower(precondition_type.name).capitalize()

                                non_precondition_runtime_array_c = [
                                    cg_non_precondition_runtime_dict[m_value][c_value][precondition_type]
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
                                        label=f"CG Un-precondition {precondition_name}",
                                        log=1)

                                precondition_runtime_array_c = [
                                    cg_precondition_runtime_dict[m_value][c_value][precondition_type]
                                    for m_value in m_array]

                                # bottom bar EXCLUDE None...
                                # green - g
                                plt.bar(bars_array[precondition_index_adjusted],
                                        precondition_runtime_array_c,
                                        width=barWidth,
                                        label=f"CG Precondition {precondition_name}",
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

            # plt.legend(loc='upper left')
            plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0)

            plt.xticks([r + offset for r in range(m_len)],
                       m_array)

            plt.grid(True, axis='y', which='both', linestyle='dotted')

            plt.savefig(f"new_figures/gmres_cg_runtime_timings_plot_c={c_value}.png", bbox_inches="tight")

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
                    for method_index, method_type in enumerate(implementation_methods_array):
                        match method_type:
                            case ImplementationMethodsEnum.GMRES:
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
                                errors_array_c_m = gmres_residual_dict[m_value][c_value][precondition_type]

                                # FIXME: just use the length of the residual (errors array) -- iteration count differs if it
                                #  stopped early
                                num_iter_residual = len(errors_array_c_m)
                                # plt.semilogy(list(range(num_iter + 1)), errors_array_c_m,
                                #              '.--',
                                #              label=method_name)
                                plt.semilogy(list(range(num_iter_residual)), errors_array_c_m,
                                             '.--',
                                             label=method_name)

                            case ImplementationMethodsEnum.CG:
                                match precondition_type:
                                    case None:
                                        method_name = "CG"
                                    case other:
                                        method_name = "CG "
                                        method_name += str.lower(other.name).capitalize()

                                # if precondition_type is None:
                                #     method_name = "GMRES"
                                # else:
                                #     method_name = str.lower(precondition_type.name).capitalize()

                                num_iter = num_iter_dict[m_value][c_value][precondition_type]
                                errors_array_c_m = cg_residual_dict[m_value][c_value][precondition_type]

                                # FIXME: just use the length of the residual (errors array) -- iteration count differs if it
                                #  stopped early
                                num_iter_residual = len(errors_array_c_m)
                                # plt.semilogy(list(range(num_iter + 1)), errors_array_c_m,
                                #              '.--',
                                #              label=method_name)
                                plt.semilogy(list(range(num_iter_residual)), errors_array_c_m,
                                             '.--',
                                             label=method_name)

                            case _:
                                raise ValueError("Invalid method type...")

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

                plt.savefig(f"new_figures/gmres_cg_residual_plot_c={c_value}_m={m_value}.png", bbox_inches="tight")


if __name__ == '__main__':
    num_max_iter = 100
    threshold = 1e-14

    m_array = [400, 800, 1600, 3200, 6400, 12800]
    # m_array = [6400, 12800]

    # starting with c = 0.75, convergence is consistent
    c_array = [0.5, 1, 10, 100, 1000]
    # c_array = [10]

    built_in_methods_array = [m_enum for m_enum in BuiltInMethodsEnum]
    implementation_methods_array = [i_m_enum for i_m_enum in ImplementationMethodsEnum]
    precondition_array = [None] + [p_enum for p_enum in PreconditionEnum]

    num_iter_dict = {}

    total_runtime_dict_true = {}
    total_runtime_dict_LU = {}

    # GMRES
    gmres_residual_dict = {}
    gmres_total_runtime_dict = {}
    gmres_non_precondition_runtime_dict = {}
    gmres_precondition_runtime_dict = {}

    # CG
    cg_residual_dict = {}
    cg_total_runtime_dict = {}
    cg_non_precondition_runtime_dict = {}
    cg_precondition_runtime_dict = {}

    print(f"len enum: {len(PreconditionEnum)}")
    print()

    for m_value in m_array:
        num_iter_dict[m_value] = {}
        total_runtime_dict_true[m_value] = {}
        total_runtime_dict_LU[m_value] = {}

        gmres_residual_dict[m_value] = {}
        gmres_total_runtime_dict[m_value] = {}
        gmres_non_precondition_runtime_dict[m_value] = {}
        gmres_precondition_runtime_dict[m_value] = {}

        cg_residual_dict[m_value] = {}
        cg_total_runtime_dict[m_value] = {}
        cg_non_precondition_runtime_dict[m_value] = {}
        cg_precondition_runtime_dict[m_value] = {}

        for c_value in c_array:
            num_iter_dict[m_value][c_value] = {}
            total_runtime_dict_true[m_value][c_value] = {}
            total_runtime_dict_LU[m_value][c_value] = {}

            gmres_residual_dict[m_value][c_value] = {}
            gmres_total_runtime_dict[m_value][c_value] = {}
            gmres_non_precondition_runtime_dict[m_value][c_value] = {}
            gmres_precondition_runtime_dict[m_value][c_value] = {}

            cg_residual_dict[m_value][c_value] = {}
            cg_total_runtime_dict[m_value][c_value] = {}
            cg_non_precondition_runtime_dict[m_value][c_value] = {}
            cg_precondition_runtime_dict[m_value][c_value] = {}

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
                print()

                for implementation_method in implementation_methods_array:
                    match implementation_method:
                        case ImplementationMethodsEnum.GMRES:
                            print(f"GMRES")

                            start_time = time()
                            x_gmres, error_list, k_final, precondition_time = \
                                gmres(A, y, num_max_iter=num_max_iter,
                                      threshold=threshold,
                                      precondition=precondition)
                            end_time = time()

                            total_time = end_time - start_time

                            num_iter_dict[m_value][c_value][precondition] = k_final
                            gmres_residual_dict[m_value][c_value][precondition] = error_list
                            gmres_total_runtime_dict[m_value][c_value][precondition] = \
                                total_time
                            gmres_precondition_runtime_dict[m_value][c_value][precondition] = \
                                precondition_time

                            non_precondition_time = total_time - precondition_time
                            gmres_non_precondition_runtime_dict[m_value][c_value][precondition] = \
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

                        case ImplementationMethodsEnum.CG:
                            print(f"Conjugate Gradient")

                            start_time = time()
                            x_cg, residual_list, k_final, precondition_time =\
                                conjugate_gradient(A, y, num_max_iter=num_max_iter, threshold=threshold,
                                                   precondition=precondition)
                            end_time = time()

                            total_time = end_time - start_time

                            # num_iter_dict[m_value][c_value][precondition] = k_final
                            cg_residual_dict[m_value][c_value][precondition] = residual_list
                            cg_total_runtime_dict[m_value][c_value][precondition] = \
                                total_time
                            cg_precondition_runtime_dict[m_value][c_value][precondition] = \
                                precondition_time

                            non_precondition_time = total_time - precondition_time
                            cg_non_precondition_runtime_dict[m_value][c_value][precondition] = \
                                non_precondition_time

                            # print(f"x true: {x_true}")
                            # print(f"x GMRES: {x_gmres}")

                            # print(f"error: {error_list}")
                            print(f"num iterations: {k_final}")
                            print(f"norm error: {np_norm(x_true - x_cg)}")
                            print(f"errors list: {residual_list}")
                            print(f"total runtime (true): {total_time_true}")
                            print(f"total runtime: {total_time}")
                            print(f"precondition time: {precondition_time}")
                            print(f"non-precondition time: {non_precondition_time}")
                            print()

    # plot the graphs
    plot_figures(m_array, c_array, built_in_methods_array, implementation_methods_array, precondition_array,
                 num_iter_dict, total_runtime_dict_true, total_runtime_dict_LU, gmres_residual_dict,
                 gmres_total_runtime_dict, gmres_precondition_runtime_dict, gmres_non_precondition_runtime_dict,
                 cg_residual_dict, cg_total_runtime_dict, cg_precondition_runtime_dict,
                 cg_non_precondition_runtime_dict, num_max_iter)

    # plot_runtime=True, plot_error=True
