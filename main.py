import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from time import time

from numpy.linalg import norm as np_norm, \
    solve as np_solve
from scipy.sparse.linalg import norm as sp_norm, spsolve
from scipy.sparse import diags as sp_diags

from gmres import gmres
from precondition import precondition_enum


def plot_figures(m_array,
                 c_array,
                 non_precondition_runtime_dict,
                 precondition_array,
                 errors_dict, total_runtime_dict_true,
                 total_runtime_dict, precondition_runtime_dict):
    # plot figures
    # Set position of bar on X axis
    m_len = len(m_array)
    c_len = len(c_array)
    precondition_len = len(precondition_array)

    num_groups = m_len
    num_bars_per_group = precondition_len
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

    # Third, add a horizontal grid (but keep the vertical grid hidden).
    # Color the lines a light gray as well.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="both", color='#EEEEEE', linestyle='dotted')
    ax.xaxis.grid(False)

    # fig.tight_layout()

    # zorder before plotting...
    # ax.set_axisbelow(True)
    # plt.grid(True, axis="y", which="both", linestyle='dotted', zorder=0)
    # plt.grid(True, axis="y", color='#EEEEEE', which="both",
    #          linestyle='dotted')
    # check if the grid looks good...

    # ax = fig.add_axes([0, 0, 1, 1])

    c_value = c_array[0]
    bottom_bar_array = []

    # for m_index, m_value in enumerate(m_array):
    for precondition_index, precondition_type in enumerate(precondition_array):
        # print(f"precondition: {precondition_type}")
        if precondition_type is None:
            total_runtime_array_c = [total_runtime_dict[m_value][c_value][precondition_type]
                                     for m_value in m_array]

            # color='...'
            # blue - b
            plt.bar(bars_array[precondition_index],
                    total_runtime_array_c,
                    width=barWidth,
                    label=f"Not Preconditioned",
                    log=1)
        else:
            precondition_name = str.lower(precondition_type.name).capitalize()

            non_precondition_runtime_array_c = [non_precondition_runtime_dict[m_value][c_value][precondition_type]
                                                for m_value in m_array]
            bottom_bar_array.append(non_precondition_runtime_array_c)
            print(non_precondition_runtime_array_c)
            print(non_precondition_runtime_dict)

            # changed from br3 -- stack the repartition time with new hybrid time
            # change so output[3] is on bottom -- new repartitioned hybrid time
            # red - r
            plt.bar(bars_array[precondition_index],
                    non_precondition_runtime_array_c,
                    width=barWidth,
                    label=f"Unpreconditioned {precondition_name}",
                    log=1)

            precondition_runtime_array_c = [precondition_runtime_dict[m_value][c_value][precondition_type]
                                            for m_value in m_array]

            # bottom bar EXCLUDE None...
            # green - g
            plt.bar(bars_array[precondition_index],
                    precondition_runtime_array_c,
                    width=barWidth,
                    label=f"Preconditioned {precondition_name}",
                    bottom=bottom_bar_array[precondition_index - 1],
                    log=1)

    ##########################
    # plot details
    plt.xlabel(f"Array Size")
    plt.ylabel(f"Time (s)")
    plt.title(f"Runtime")

    offset = ((1 / 2) * (num_bars_per_group - 1)) * barWidth

    plt.legend()

    plt.xticks([r + offset for r in range(m_len)],
               m_array)

    plt.savefig(f"test_runtime_timings_plot.png", bbox_inches="tight")

    print(bottom_bar_array)


if __name__ == '__main__':
    # m_array = [600, 1300, 2500, 5000, 1000, 2000]
    # m_array = [600, 1200, 2400, 4800, 9600, 19200]
    # m_array = [400, 800, 1600, 3200, 6400, 12800]
    m_array = [6400, 12800]

    a = []
    b = []
    for i in range(3):
        b = [1, 2, 3]
        b.append(i)
        a.append(b)
    print(a)

    # b_array = [10, 50, 100, 500, 1000, 5000, 10000]
    # b_array = [1]

    # starting with c = 0.75, convergence is consistent
    # c_array = [0.5, 1, 2, 4, 8]
    c_array = [2]

    # TODO: do
    # precondition_array = [precondition_enum.JACOBI,
    #                       precondition_enum.GAUSS_SEIDEL,
    #                       precondition_enum.SYMMETRIC_GAUSS_SEIDEL]
    # precondition_array = [precondition_enum.name
    #                       for p_enum in precondition_enum]
    precondition_array = [None] + [p_enum for p_enum in precondition_enum]

    errors_dict = {}
    total_runtime_dict_true = {}
    total_runtime_dict = {}
    non_precondition_runtime_dict = {}
    precondition_runtime_dict = {}

    print(f"len enum: {len(precondition_enum)}")

    for m_value in m_array:
        errors_dict[m_value] = {}
        total_runtime_dict_true[m_value] = {}
        total_runtime_dict[m_value] = {}
        non_precondition_runtime_dict[m_value] = {}
        precondition_runtime_dict[m_value] = {}

        for c_value in c_array:
            errors_dict[m_value][c_value] = {}
            total_runtime_dict_true[m_value][c_value] = {}
            total_runtime_dict[m_value][c_value] = {}
            non_precondition_runtime_dict[m_value][c_value] = {}
            precondition_runtime_dict[m_value][c_value] = {}

            print(f"m: {m_value}")
            print(f"c: {c_value}")

            for precondition in precondition_array:
                print(f"precondition: {precondition}")

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

                total_time_true = end_time_true - start_time_true

                start_time = time()
                x_gmres, error_list, k_final, precondition_time = \
                    gmres(A, y, precondition=precondition)
                end_time = time()

                total_time = end_time - start_time

                errors_dict[m_value][c_value][precondition] = error_list
                total_runtime_dict_true[m_value][c_value][precondition] = \
                    total_time_true
                total_runtime_dict[m_value][c_value][precondition] = \
                    total_time
                precondition_runtime_dict[m_value][c_value][precondition] = \
                    precondition_time
                non_precondition_runtime_dict[m_value][c_value][precondition] = \
                    total_time - precondition_time

                # print(f"x true: {x_true}")
                # print(f"x GMRES: {x_gmres}")

                # print(f"error: {error_list}")
                print(f"num iterations: {k_final}")
                print(f"norm error: {np_norm(x_true - x_gmres)}")

    # plot the graphs
    plot_figures(m_array, c_array, non_precondition_runtime_dict, precondition_array, errors_dict,
                 total_runtime_dict_true, total_runtime_dict, precondition_runtime_dict)
