from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
from scipy.stats import linregress
import os
import convergence.exact_solution as sol
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")


def compute_error(computed_sol, analytical_sol, var_name, norm_type="L2"):
    # Now we compute the various metrics. First we
    # simply compute the L2 error between the analytic
    # solutions and the computed ones.
    error = errornorm(analytical_sol, computed_sol, norm_type=norm_type)

    # We keep track of all metrics using a Python dictionary
    error_dictionary = {var_name: error}

    return error_dictionary


def convergence_hp(
    solver,
    min_degree=1,
    max_degree=4,
    numel_xy=[2, 4, 8, 16, 32, 64, 128, 256],
    norm_type='L2',
    quadrilateral=True,
    name='',
    **kwargs
):
    if name:
        name += '_'
    for degree in range(min_degree, max_degree):
        p1_errors = np.array([])
        p2_errors = np.array([])
        v1_errors = np.array([])
        v2_errors = np.array([])
        num_cells = np.array([])
        mesh_size = np.array([])
        for n in numel_xy:
            nel_x = nel_y = n
            mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=quadrilateral)
            num_cells = np.append(num_cells, mesh.num_cells())
            mesh_size = np.append(mesh_size, 1. / n)

            p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2 = solver(mesh=mesh, degree=degree, **kwargs)
            error_dictionary = {}
            error_dictionary.update(compute_error(p1_sol, p_e_1, 'p1_error', norm_type=norm_type))
            error_dictionary.update(compute_error(p2_sol, p_e_2, 'p2_error', norm_type=norm_type))
            p1_errors = np.append(p1_errors, error_dictionary['p1_error'])
            p2_errors = np.append(p2_errors, error_dictionary['p2_error'])
            error_dictionary.update(compute_error(v1_sol, v_e_1, 'v1_error'), norm_type=norm_type)
            error_dictionary.update(compute_error(v2_sol, v_e_2, 'v2_error'), norm_type=norm_type)
            v1_errors = np.append(v1_errors, error_dictionary['v1_error'])
            v2_errors = np.append(v2_errors, error_dictionary['v2_error'])
        p1_errors_log2 = np.log2(p1_errors)
        p2_errors_log2 = np.log2(p2_errors)
        v1_errors_log2 = np.log2(v1_errors)
        v2_errors_log2 = np.log2(v2_errors)
        num_cells_log2 = np.log2(num_cells)
        mesh_size_log2 = np.log2(mesh_size)
        p1_slope, intercept1, r_value1, p_value1, stderr1 = linregress(mesh_size_log2, p1_errors_log2)
        p2_slope, intercept2, r_value2, p_value2, stderr2 = linregress(mesh_size_log2, p2_errors_log2)
        PETSc.Sys.Print(
            "\n--------------------------------------\nDegree %d: p1 slope error %f" % (degree, np.abs(p1_slope)),
            "\nDegree %d: p2 slope error %f" % (degree, np.abs(p2_slope)),
        )
        v1_slope, intercept_v1, r_value_v1, p_value_v1, stderr_v1 = linregress(mesh_size_log2, v1_errors_log2)
        v2_slope, intercept_v2, r_value_v2, p_value_v2, stderr_v2 = linregress(mesh_size_log2, v2_errors_log2)
        PETSc.Sys.Print(
            "\n--------------------------------------\nDegree %d: v1 slope error %f" % (degree, np.abs(v1_slope)),
            "\nDegree %d: v2 slope error %f" % (degree, np.abs(v2_slope)),
        )
        # _plot_errors(mesh_size, p1_errors, p1_slope, degree, name='p1_errors')
        # _plot_errors(mesh_size, p2_errors, p2_slope, degree, name='p2_errors')
        # _plot_errors(mesh_size, v1_errors, v1_slope, degree, name='v1_errors')
        # _plot_errors(mesh_size, v2_errors, v2_slope, degree, name='v2_errors')
        os.makedirs("results_%s" % name, exist_ok=True)
        np.savetxt(
            ('results_%s/errors_degree%d.dat' % (name, degree)),
            np.transpose([-mesh_size_log2, p1_errors_log2, p2_errors_log2, v1_errors_log2, v2_errors_log2])
        )

    return


def _plot_errors(mesh_size, errors, slope, degree, name='Error'):
    plt.figure()
    plt.loglog(mesh_size, errors, '-o', label=(r'k = %d; slope = %f' % (degree, np.abs(slope))))
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('%s_deg%i.png' % (name, degree))
    return
