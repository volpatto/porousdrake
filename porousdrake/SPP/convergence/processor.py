from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
import pandas as pd
from scipy.stats import linregress
import os
import porousdrake.SPP.convergence.exact_solution as sol

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except ImportError:
    warning("Matplotlib not imported")


def convergence_hp(
    solver,
    min_degree=1,
    max_degree=4,
    numel_xy=(2, 4, 8, 16, 32, 64, 128, 256),
    quadrilateral=True,
    name="",
    **kwargs
):
    computed_errors_dict = {
        "Element": list(),
        "Degree": list(),
        "Cells": list(),
        "Mesh size": list(),
        "L2-error p": list(),
        "H1-error p": list(),
        "L2-error u": list(),
        "Hdiv-error u": list(),
    }
    element_kind = "Quad" if quadrilateral else "Tri"
    for degree in range(min_degree, max_degree):
        p_errors = np.array([])
        p_errors_h1 = np.array([])
        v_errors = np.array([])
        v_errors_hdiv = np.array([])
        num_cells = np.array([])
        mesh_size = np.array([])
        for n in numel_xy:
            nel_x = nel_y = n
            mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=quadrilateral)
            current_num_cells = mesh.num_cells()
            num_cells = np.append(num_cells, current_num_cells)
            current_mesh_size = mesh.cell_sizes.dat.data_ro.min()
            mesh_size = np.append(mesh_size, current_mesh_size)

            p_sol, v_sol, p_e, v_e = solver(mesh=mesh, degree=degree, **kwargs)

            current_error_p = errornorm(p_sol, p_e, norm_type="L2")
            p_errors = np.append(p_errors, current_error_p)

            current_error_p_h1 = errornorm(p_sol, p_e, norm_type="H1")
            p_errors_h1 = np.append(p_errors_h1, current_error_p_h1)

            current_error_v = errornorm(v_sol, v_e, norm_type="L2")
            v_errors = np.append(v_errors, current_error_v)

            current_error_v_hdiv = errornorm(v_sol, v_e, norm_type="Hdiv")
            v_errors_hdiv = np.append(v_errors_hdiv, current_error_v_hdiv)

            computed_errors_dict["Element"].append(element_kind)
            computed_errors_dict["Degree"].append(degree)
            computed_errors_dict["Cells"].append(current_num_cells)
            computed_errors_dict["Mesh size"].append(current_mesh_size)
            computed_errors_dict["L2-error p"].append(current_error_p)
            computed_errors_dict["H1-error p"].append(current_error_p)
            computed_errors_dict["L2-error u"].append(current_error_v)
            computed_errors_dict["Hdiv-error u"].append(current_error_v_hdiv)

        p_errors_log10 = np.log10(p_errors)
        v_errors_log10 = np.log10(v_errors)
        num_cells_log10 = np.log10(num_cells)
        mesh_size_log10 = np.log10(mesh_size)
        p_slope, intercept, r_value, p_value, stderr = linregress(mesh_size_log10, p_errors_log10)
        PETSc.Sys.Print(
            "\n--------------------------------------\nDegree %d: p slope error %f"
            % (degree, np.abs(p_slope))
        )
        v_slope, intercept_v, r_value_v, p_value_v, stderr_v = linregress(
            mesh_size_log10, v_errors_log10
        )
        PETSc.Sys.Print(
            "\n--------------------------------------\nDegree %d: v slope error %f"
            % (degree, np.abs(v_slope))
        )

    os.makedirs("results_%s" % name, exist_ok=True)
    df_computed_errors = pd.DataFrame(data=computed_errors_dict)
    path_to_save_results = "results_%s/errors.csv" % name
    df_computed_errors.to_csv(path_to_save_results)

    return


def _plot_errors(mesh_size, errors, slope, degree, name="Error"):
    plt.figure()
    plt.loglog(mesh_size, errors, "-o", label=(r"k = %d; slope = %f" % (degree, np.abs(slope))))
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("%s_deg%i.png" % (name, degree))
    return
