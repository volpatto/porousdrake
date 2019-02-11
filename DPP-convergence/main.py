from firedrake import *
import solvers
import convergence
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 4, 4
Lx, Ly = 1.0, 1.0
quadrilateral = True
degree = 1
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Cold run
p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2 = solvers.sdhm(
    mesh=mesh,
    degree=degree,
    delta_0=Constant(1),
    delta_1=Constant(-0.5),
    delta_2=Constant(0.5),
    delta_3=Constant(0.5),
    beta_0=Constant(1e-10)
)
print('*** Cold run OK ***\n')

convergence.convergence_hp(
    solvers.cgls,
    max_degree=3,
    mesh_pow_min=3,
    mesh_pow_max=7,
    quadrilateral=True,
    delta_0=Constant(-1),
    delta_1=Constant(0.5),
    delta_2=Constant(0.0),
    delta_3=Constant(0.0)
)
