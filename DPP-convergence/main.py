from firedrake import *
import sdhm
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
p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2 = sdhm.sdhm(
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
    max_degree=3,
    delta_0=Constant(1),
    delta_1=Constant(-0.5),
    delta_2=Constant(0.5),
    delta_3=Constant(0.5),
    beta_0=Constant(1e-10)
)
