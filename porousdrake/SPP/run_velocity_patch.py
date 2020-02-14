from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import os
import sys

from porousdrake.SPP.velocity_patch.solvers import cgls, dgls, sdhm, lsh
import porousdrake.setup.solvers_parameters as parameters

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except ImportError:
    warning("Matplotlib not imported")

single_evaluation = True
nx, ny = 50, 30
Lx, Ly = 5.0, 4.0
quadrilateral = True
degree = 1
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Solver options
solvers_options = {
    "cgls_full": cgls,
    "cgls_div": cgls,
    "mgls": cgls,
    "mvh_full": cgls,
    "mvh_div": cgls,
    "mvh": cgls,
    "dgls_full": dgls,
    "dmgls_full": dgls,
    "dmvh_full": dgls,
    "sdhm_full": sdhm,
    "sdhm_div": sdhm,
    "hmvh_full": sdhm,
    "hmvh": sdhm,
    "lsh": lsh,
}

# Identify discontinuous solvers for writing .pvd purpose
discontinuous_solvers = ["dgls_full", "dmgls_full", "dmvh_full", "sdhm_full", "hmvh_full", "hmvh"]

if single_evaluation:

    # Choosing the solver
    selected_solver = "sdhm_full"
    solver = solvers_options[selected_solver]
    solver_kwargs = parameters.solvers_args[selected_solver]

    p_sol, v_sol = solver(
        mesh=mesh, degree=degree, mesh_parameter=parameters.mesh_parameter, **solver_kwargs
    )
    plot(v_sol.sub(0))
    plt.show()

    PETSc.Sys.Print(f"*** Single running done for {selected_solver} ***")
    sys.exit(0)

# Sanity check for keys among solvers_options and solvers_args
assert set(solvers_options.keys()).issubset(parameters.solvers_args.keys())

# Solving velocity patch for the selected methods
os.makedirs("velocity_patch/output", exist_ok=True)
output_file_1 = File("velocity_patch/output/continuous_velocity_solutions.pvd")
output_file_2 = File("velocity_patch/output/discontinuous_velocity_solutions.pvd")
continuous_solutions = []
discontinuous_solutions = []
for current_solver in solvers_options:
    PETSc.Sys.Print("*******************************************\n")
    PETSc.Sys.Print("*** Begin case: %s ***\n" % current_solver)

    # Selecting the solver and its kwargs
    solver = solvers_options[current_solver]
    kwargs = parameters.solvers_args[current_solver]

    # Appending the mesh parameter option to kwargs
    kwargs["mesh_parameter"] = True

    # Running the case
    current_solution = solver(mesh=mesh, degree=degree, **kwargs)

    # Renaming to identify the velocities properly
    current_solution[1].rename("v_x (%s)" % current_solver, "label")

    # Appending the solution
    if current_solver in discontinuous_solvers:
        discontinuous_solutions.append(current_solution[1].sub(0))
    else:
        continuous_solutions.append(current_solution[1].sub(0))

    # plot(current_solution[1].sub(0))
    # plot(current_solution[3].sub(0))
    # plt.show()

    PETSc.Sys.Print("\n*** End case: %s ***" % current_solver)
    PETSc.Sys.Print("*******************************************\n")

# Writing in the .pvd file
output_file_1.write(*continuous_solutions)
output_file_2.write(*discontinuous_solutions)
