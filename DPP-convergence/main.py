from firedrake import *
import solvers
import convergence
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 2**3, 2**3
Lx, Ly = 1.0, 1.0
quadrilateral = True
degree = 1
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Stabilizing parameters
delta_0 = Constant(1)
delta_1 = Constant(-0.5)
delta_2 = Constant(0.5)
delta_3 = Constant(0.5)
eta_u = Constant(10.0)
eta_p = 100 * eta_u
beta_0 = Constant(1.0)

# Choosing the solver
solver = solvers.sdhm

# Cold run
p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2 = solver(
    mesh=mesh,
    degree=degree,
    delta_0=delta_0,
    delta_1=delta_1,
    delta_2=delta_2,
    delta_3=delta_3,
    beta_0=beta_0
)
plot(p1_sol)
plot(p_e_1)
plot(p2_sol)
plot(p_e_2)
plot(v1_sol)
plot(v_e_1)
plot(v2_sol)
plot(v_e_2)
plt.show()
print('*** Cold run OK ***\n')

convergence.convergence_hp(
    solver,
    min_degree=degree,
    max_degree=degree + 1,
    mesh_pow_min=3,
    mesh_pow_max=7,
    quadrilateral=quadrilateral,
    delta_0=delta_0,
    delta_1=delta_1,
    delta_2=delta_2,
    delta_3=delta_3,
    beta_0=beta_0
)
