from firedrake import *
from porousdrake.SPP.velocity_patch.solvers import cgls, dgls, sdhm
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import sys
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
Lx, Ly = 5.0, 4.0
quadrilateral = True
degree = 1
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Stabilizing parameters
delta_0 = Constant(1)
delta_1 = Constant(-0.5)
delta_2 = Constant(0.5)
delta_3 = Constant(0.5)
eta_u = Constant(10.0)
eta_p = 10 * eta_u
beta_0 = Constant(1.0e-15)
mesh_parameter = True

# Choosing the solver
solver = sdhm

p_sol, v_sol = solver(
    mesh=mesh,
    degree=degree,
    delta_0=delta_0,
    delta_1=delta_1,
    delta_2=delta_2,
    delta_3=delta_3,
    beta_0=beta_0,
    # eta_u=eta_u,
    # eta_p=eta_p,
    mesh_parameter=mesh_parameter
)
plot(v_sol.sub(0))
plt.show()
# pp.write_pvd_mixed_formulations('teste_nohup', mesh, degree, p1_sol, v1_sol, p2_sol, v2_sol)
