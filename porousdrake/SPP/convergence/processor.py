from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
from scipy.stats import linregress
import os
import porousdrake.SPP.convergence.exact_solution as sol

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
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
    numel_xy=(2, 4, 8, 16, 32, 64, 128, 256),
    norm_type="L2",
    quadrilateral=True,
    name="",
    **kwargs
):
    for degree in range(min_degree, max_degree):
        p_errors = np.array([])
        v_errors = np.array([])
        num_cells = np.array([])
        mesh_size = np.array([])
        for n in numel_xy:
            nel_x = nel_y = n
            mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=quadrilateral)
            num_cells = np.append(num_cells, mesh.num_cells())
            mesh_size = np.append(mesh_size, 1.0 / n)

            p_sol, v_sol, p_e, v_e = solver(mesh=mesh, degree=degree, **kwargs)
            error_dictionary = {}
            error_dictionary.update(compute_error(p_sol, p_e, "p_error", norm_type=norm_type))
            p_errors = np.append(p_errors, error_dictionary["p_error"])
            error_dictionary.update(compute_error(v_sol, v_e, "v_error"), norm_type=norm_type)
            v_errors = np.append(v_errors, error_dictionary["v_error"])
        p_errors_log2 = np.log2(p_errors)
        v_errors_log2 = np.log2(v_errors)
        num_cells_log2 = np.log2(num_cells)
        mesh_size_log2 = np.log2(mesh_size)
        p_slope, intercept, r_value, p_value, stderr = linregress(mesh_size_log2, p_errors_log2)
        PETSc.Sys.Print(
            "\n--------------------------------------\nDegree %d: p slope error %f"
            % (degree, np.abs(p_slope))
        )
        v_slope, intercept_v, r_value_v, p_value_v, stderr_v = linregress(
            mesh_size_log2, v_errors_log2
        )
        PETSc.Sys.Print(
            "\n--------------------------------------\nDegree %d: v slope error %f"
            % (degree, np.abs(v_slope))
        )
        # _plot_errors(mesh_size, p1_errors, p1_slope, degree, name='p1_errors')
        # _plot_errors(mesh_size, p2_errors, p2_slope, degree, name='p2_errors')
        # _plot_errors(mesh_size, v1_errors, v1_slope, degree, name='v1_errors')
        # _plot_errors(mesh_size, v2_errors, v2_slope, degree, name='v2_errors')
        os.makedirs("results_%s" % name, exist_ok=True)
        np.savetxt(
            ("results_%s/errors_degree%d.dat" % (name, degree)),
            np.transpose([-mesh_size_log2, p_errors_log2, v_errors_log2]),
        )

    return


def _plot_errors(mesh_size, errors, slope, degree, name="Error"):
    plt.figure()
    plt.loglog(mesh_size, errors, "-o", label=(r"k = %d; slope = %f" % (degree, np.abs(slope))))
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("%s_deg%i.png" % (name, degree))
    return
