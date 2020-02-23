from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

from porousdrake.SPP.convergence import processor
from porousdrake.SPP.convergence.solvers import cgls, dgls, sdhm, lsh, dls, clsq
from porousdrake.setup import solvers_parameters as parameters

import porousdrake.post_processing.writers as pp
import sys
import os

temp_dir_to_save_results = "./results_temp"
os.makedirs(temp_dir_to_save_results, exist_ok=True)

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except ImportError:
    warning("Matplotlib not imported")

single_run = False
single_run_plot = False
single_run_write_results = False
nx, ny = 10, 10
Lx, Ly = 1.0, 1.0
quadrilateral = False
degree = 1
last_degree = 4
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Mesh options
mesh_quad = [False, True]  # Triangles, Quads
# mesh_parameters = [True, False]
mesh_parameters = [True]

# Solver options
solvers_options = {
    # "cgls_full": cgls,
    # "cgls_div": cgls,
    # "mgls": cgls,
    # "mgls_full": cgls,
    # "mvh_full": cgls,
    # "mvh_div": cgls,
    # "mvh": cgls,
    # "dgls_full": dgls,
    # "dgls_div": dgls,
    # "dmgls": dgls,
    # "dmgls_full": dgls,
    # "dmvh_full": dgls,
    # "dmvh_div": dgls,
    # "dmvh": dgls,
    # "sdhm_full": sdhm,
    # "sdhm_div": sdhm,
    # "hmgls": sdhm,
    # "hmgls_full": sdhm,
    # "hmvh_full": sdhm,
    # "hmvh_div": sdhm,
    # "hmvh": sdhm,
    # "lsh": lsh,
    # "lsh_mass": lsh,
    # "lsh_lambda": lsh,
    # "lsh_full": lsh,
    "dls": dls,
    # "clsq": clsq,
}

# Convergence range
n = [5, 10, 15, 20, 25, 30]
# n = [10, 15, 20, 25, 30, 35]
# n = [4, 8, 16, 32, 64, 128]

# Cold run
if single_run:

    # Choosing the solver
    selected_solver = "dls"
    solver = solvers_options[selected_solver]
    solver_kwargs = parameters.solvers_args[selected_solver]

    p_sol, v_sol, p_e, v_e = solver(
        mesh=mesh, degree=degree, mesh_parameter=parameters.mesh_parameter, **solver_kwargs
    )

    if single_run_plot:
        plot(p_sol)
        plt.show()

        plot(p_e)
        plt.show()

        plot(v_sol)
        plt.show()

        plot(v_e)
        plt.show()

    if single_run_write_results:
        p_sol.rename("%s_p" % selected_solver)
        p_e.rename("exact_p")
        v_sol.rename("%s_v" % selected_solver)
        v_e.rename("exact_v")
        outfile = File("%s/%s.pvd" % (temp_dir_to_save_results, selected_solver))
        outfile.write(p_sol, v_sol, p_e, v_e)

    print("*** Cold run OK ***\n")
    sys.exit()

# Sanity check for keys among solvers_options and solvers_args
assert set(solvers_options.keys()).issubset(parameters.solvers_args.keys())

for element in mesh_quad:
    for current_solver in solvers_options:
        for mesh_parameter in mesh_parameters:
            if element:
                element_kind = "quad"
            else:
                element_kind = "tri"
            if mesh_parameter:
                mesh_par = ""
            else:
                mesh_par = "meshless_par"

            # Setting the output file name
            name = "%s_%s_%s_errors" % (current_solver, mesh_par, element_kind)
            PETSc.Sys.Print("*******************************************\n")
            PETSc.Sys.Print("*** Begin case: %s ***\n" % name)

            # Selecting the solver and its kwargs
            solver = solvers_options[current_solver]
            kwargs = parameters.solvers_args[current_solver]

            # Appending the mesh parameter option to kwargs
            kwargs["mesh_parameter"] = mesh_parameter

            # Performing the convergence study
            processor.convergence_hp(
                solver,
                min_degree=degree,
                max_degree=degree + last_degree,
                numel_xy=n,
                quadrilateral=element,
                name=name,
                **kwargs,
            )
            PETSc.Sys.Print("\n*** End case: %s ***" % name)
            PETSc.Sys.Print("*******************************************\n")
