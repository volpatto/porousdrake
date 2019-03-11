from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

from porousdrake.DPP.velocity_patch.solvers import cgls, dgls, sdhm
import porousdrake.setup.solvers_parameters as parameters

try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 10, 10
Lx, Ly = 5.0, 4.0
quadrilateral = True
degree = 1
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Solver options
solvers_options = {
    'cgls_full': cgls,
    'mgls_full': cgls,
    'mvh_full': cgls,
    'dgls_full': dgls,
    'dmgls_full': dgls,
    'dmvh_full': dgls,
    'sdhm_full': sdhm,
    'hmvh_full': sdhm,
    'hmvh': sdhm
}

# Choosing the solver
solver = cgls

p1_sol, v1_sol, p2_sol, v2_sol = solver(
    mesh=mesh,
    degree=degree,
    delta_0=parameters.delta_0,
    delta_1=parameters.delta_1,
    delta_2=parameters.delta_2,
    delta_3=parameters.delta_3,
    # beta_0=parameters.beta_0,
    # eta_u=parameters.eta_u,
    # eta_p=parameters.eta_p,
    mesh_parameter=parameters.mesh_parameter
)
plot(v1_sol.sub(0))
plot(v2_sol.sub(0))
plt.show()
# pp.write_pvd_mixed_formulations('teste_nohup', mesh, degree, p1_sol, v1_sol, p2_sol, v2_sol)
