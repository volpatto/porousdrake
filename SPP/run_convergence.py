from firedrake import *
from convergence.solvers import cgls, dgls, sdhm
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
from convergence import processor
import postprocessing as pp
import sys
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

single_run = True
nx, ny = 10, 10
Lx, Ly = 1.0, 1.0
quadrilateral = True
degree = 1
last_degree = 4
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Mesh options
mesh_quad = [False, True]  # Triangles, Quads
mesh_parameters = [True, False]

# Solver options
solvers_options = {
    'cgls_full': cgls,
    'cgls_div': cgls,
    'mgls': cgls,
    'mgls_full': cgls,
    'mvh_full': cgls,
    'mvh_div': cgls,
    'mvh': cgls,
    'dgls_full': dgls,
    'dgls_div': dgls,
    'dmgls': dgls,
    'dmgls_full': dgls,
    'dmvh_full': dgls,
    'dmvh_div': dgls,
    'dmvh': dgls,
    'sdhm_full': sdhm,
    'sdhm_div': sdhm,
    'hmgls': sdhm,
    'hmgls_full': sdhm,
    'hmvh_full': sdhm,
    'hmvh_div': sdhm,
    'hmvh': sdhm
}

# Stabilizing parameters
delta_0 = Constant(1)
delta_1 = Constant(-0.5)
delta_2 = Constant(0.5)
delta_3 = Constant(0.5)
eta_u = Constant(100.0)
eta_p = 1 * eta_u
beta_0 = Constant(1.0e-15)
mesh_parameter = True

# Choosing the solver
solver = cgls

# Convergence range
n = [5, 10, 15, 20, 25, 30]
#n = [4, 8, 16, 32, 64, 128]

# Cold run
if single_run:
    p_sol, v_sol, p_e, v_e = solver(
        mesh=mesh,
        degree=degree,
        delta_0=delta_0,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=delta_3,
        # beta_0=beta_0,
        # eta_u=eta_u,
        # eta_p=eta_p,
        mesh_parameter=mesh_parameter
    )

    plot(p_sol)
    plot(p_e)
    plot(v_sol)
    plot(v_e)
    plt.show()
    print('*** Cold run OK ***\n')
    # pp.write_pvd_mixed_formulations('teste_nohup', mesh, degree, p1_sol, v1_sol, p2_sol, v2_sol)
    sys.exit()

solvers_kwargs = {
    'cgls_full': {
        'delta_0': Constant(1),
        'delta_1': Constant(-0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5)
    },
    'cgls_div': {
        'delta_0': Constant(1),
        'delta_1': Constant(-0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0)
    },
    'mgls_full': {
        'delta_0': Constant(1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5)
    },
    'mgls': {
        'delta_0': Constant(1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0)
    },
    'mvh_full': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5)
    },
    'mvh_div': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0)
    },
    'mvh': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.0),
        'delta_3': Constant(0.0)
    },
    ###############################################
    'dgls_full': {
        'delta_0': Constant(1),
        'delta_1': Constant(-0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    'dgls_div': {
        'delta_0': Constant(1),
        'delta_1': Constant(-0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    'dmgls_full': {
        'delta_0': Constant(1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    'dmgls': {
        'delta_0': Constant(1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    'dmvh_full': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    'dmvh_div': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    'dmvh': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.0),
        'delta_3': Constant(0.0),
        'eta_u': eta_u,
        'eta_p': eta_p
    },
    ###############################################
    'sdhm_full': {
        'delta_0': Constant(1),
        'delta_1': Constant(-0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5),
        'beta_0': beta_0
    },
    'sdhm_div': {
        'delta_0': Constant(1),
        'delta_1': Constant(-0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0),
        'beta_0': beta_0
    },
    'hmgls_full': {
        'delta_0': Constant(1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5),
        'beta_0': beta_0
    },
    'hmgls': {
        'delta_0': Constant(1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0),
        'beta_0': beta_0
    },
    'hmvh_full': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.5),
        'beta_0': beta_0
    },
    'hmvh_div': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.5),
        'delta_3': Constant(0.0),
        'beta_0': beta_0
    },
    'hmvh': {
        'delta_0': Constant(-1),
        'delta_1': Constant(0.5),
        'delta_2': Constant(0.0),
        'delta_3': Constant(0.0),
        'beta_0': beta_0
    },
}

# Sanity check for keys among solvers_options and solvers_kwargs
assert set(solvers_options.keys()).issubset(solvers_kwargs.keys())

for element in mesh_quad:
    for current_solver in solvers_options:
        for mesh_parameter in mesh_parameters:
            if element:
                element_kind = 'quad'
            else:
                element_kind = 'tri'
            if mesh_parameter:
                mesh_par = ''
            else:
                mesh_par = 'meshless_par'

            # Setting the output file name
            name = '%s_%s_%s_errors' % (current_solver, mesh_par, element_kind)
            PETSc.Sys.Print("*******************************************\n")
            PETSc.Sys.Print("*** Begin case: %s ***\n" % name)

            # Selecting the solver and its kwargs
            solver = solvers_options[current_solver]
            kwargs = solvers_kwargs[current_solver]

            # Appending the mesh parameter option to kwargs
            kwargs['mesh_parameter'] = mesh_parameter

            # Performing the convergence study
            processor.convergence_hp(
                solver,
                min_degree=degree,
                max_degree=degree + last_degree,
                numel_xy=n,
                quadrilateral=quadrilateral,
                name=name,
                **kwargs
            )
            PETSc.Sys.Print("\n*** End case: %s ***" % name)
            PETSc.Sys.Print("*******************************************\n")
