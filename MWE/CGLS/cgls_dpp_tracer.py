from firedrake import *
import numpy as np
import random
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import os

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")

random.seed(222)
nx, ny = 50, 20
Lx, Ly = 1.0, 0.4
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis("off")

degree = 1
pressure_family = "CG"
velocity_family = "CG"
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = U * V * U * V

# Trial and test functions
DPP_solution = Function(W)
u1, p1, u2, p2 = TrialFunctions(W)
v1, q1, v2, q2 = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
h = CellDiameter(mesh)

#################################################
# *** Model parameters
#################################################
uSpace = FunctionSpace(mesh, "CG", 1)
kSpace = FunctionSpace(mesh, "DG", 0)

mu0, Rc, D = Constant(1e-3), Constant(3.0), Constant(2e-6)
tol = 1e-14

k1_0 = 1.1
k1_1 = 0.9


class myk1(Expression):
    def eval(self, values, x):
        if x[1] < Ly / 2.0 + tol:
            values[0] = k1_0
        else:
            values[0] = k1_1


k1 = interpolate(myk1(), kSpace)

k2_0 = 0.01 * k1_0
k2_1 = 0.01 * k1_1


class myk2(Expression):
    def eval(self, values, x):
        if x[1] < Ly / 2.0 + tol:
            values[0] = k2_0
        else:
            values[0] = k2_1


k2 = interpolate(myk2(), kSpace)


def alpha1(c):
    return mu0 * exp(Rc * (1.0 - c)) / k1


def invalpha1(c):
    return 1.0 / alpha1(c)


def alpha2(c):
    return mu0 * exp(Rc * (1.0 - c)) / k2


def invalpha2(c):
    return 1.0 / alpha2(c)


class c_0(Expression):
    def eval(self, values, x):
        if x[0] < 0.010 * Lx:
            values[0] = abs(0.1 * exp(-x[0] * x[0]) * random.random())
        else:
            values[0] = 0.0


c1 = TrialFunction(uSpace)
u = TestFunction(uSpace)
conc = Function(uSpace)
conc_k = interpolate(c_0(), uSpace)

#################################################
#################################################
#################################################

# BCs
p_L = Constant(10.0)
p_R = Constant(1.0)
bcDPP = []
c_inj = Constant(1.0)
bcleft_c = DirichletBC(uSpace, c_inj, 1)
bcAD = [bcleft_c]
eta_u = Constant(50)

# Time parameters
T = 1.5e-3
dt = 5e-5

# Source term
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

# Stabilizing parameter
delta_0 = Constant(-1.0)
delta_1 = Constant(0.5)
delta_2 = h * h * Constant(0.5)
delta_3 = h * h * Constant(0.5)

# Mixed classical terms
a = (dot(alpha1(conc_k) * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
a += (dot(alpha2(conc_k) * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
a += delta_0 * q1 * (invalpha1(conc_k) / k1) * (p2 - p1) * dx
a += delta_0 * q2 * (invalpha2(conc_k) / k2) * (p1 - p2) * dx
L = -delta_0 * dot(rhob1, v1) * dx
L += -delta_0 * dot(rhob2, v2) * dx
# Stabilizing terms
###
a += (
    delta_1
    * inner(
        invalpha1(conc_k) * (alpha1(conc_k) * u1 + grad(p1)),
        delta_0 * alpha1(conc_k) * v1 + grad(q1),
    )
    * dx
)
a += (
    delta_1
    * inner(
        invalpha2(conc_k) * (alpha2(conc_k) * u2 + grad(p2)),
        delta_0 * alpha2(conc_k) * v2 + grad(q2),
    )
    * dx
)
###
a += delta_2 * alpha1(conc_k) * div(u1) * div(v1) * dx
a += delta_2 * alpha2(conc_k) * div(u2) * div(v2) * dx
a += -delta_2 * alpha1(conc_k) * (invalpha1(conc_k) / k1) * (p2 - p1) * div(v1) * dx
a += -delta_2 * alpha2(conc_k) * (invalpha2(conc_k) / k2) * (p1 - p2) * div(v2) * dx
###
a += delta_3 * inner(invalpha1(conc_k) * curl(alpha1(conc_k) * u1), curl(alpha1(conc_k) * v1)) * dx
a += delta_3 * inner(invalpha2(conc_k) * curl(alpha2(conc_k) * u2), curl(alpha2(conc_k) * v2)) * dx
# Weakly imposed BC
a += (
    dot(v1, n) * p1 * ds(3)
    + dot(v2, n) * p2 * ds(3)
    - q1 * dot(u1, n) * ds(3)
    - q2 * dot(u2, n) * ds(3)
    + dot(v1, n) * p1 * ds(4)
    + dot(v2, n) * p2 * ds(4)
    - q1 * dot(u1, n) * ds(4)
    - q2 * dot(u2, n) * ds(4)
)
L += (
    -dot(v1, n) * p_L * ds(1)
    - dot(v2, n) * p_L * ds(1)
    - dot(v1, n) * p_R * ds(2)
    - dot(v2, n) * p_R * ds(2)
)
a += eta_u / h * inner(dot(v1, n), dot(u1, n)) * (ds(3) + ds(4)) + eta_u / h * inner(
    dot(v2, n), dot(u2, n)
) * (ds(3) + ds(4))

# *** Transport problem
vnorm = sqrt(
    dot((DPP_solution.sub(0) + DPP_solution.sub(2)), (DPP_solution.sub(0) + DPP_solution.sub(2)))
)

taw = h / (2.0 * vnorm) * dot((DPP_solution.sub(0) + DPP_solution.sub(2)), grad(u))

a_r = (
    taw
    * (c1 + dt * (dot((DPP_solution.sub(0) + DPP_solution.sub(2)), grad(c1)) - div(D * grad(c1))))
    * dx
)

L_r = taw * (conc_k + dt * f) * dx

aAD = (
    a_r
    + u * c1 * dx
    + dt
    * (
        u * dot((DPP_solution.sub(0) + DPP_solution.sub(2)), grad(c1)) * dx
        + dot(grad(u), D * grad(c1)) * dx
    )
)
LAD = L_r + u * conc_k * dx + dt * u * f * dx

save_path = "results_%s" % os.path.basename(__file__)[:-3]
os.makedirs(save_path, exist_ok=True)

cfile = File(save_path + "/Concentration.pvd")
v1file = File(save_path + "/Macro_Velocity.pvd")
p1file = File(save_path + "/Macro_Pressure.pvd")
v2file = File(save_path + "/Micro_Velocity.pvd")
p2file = File(save_path + "/Micro_Pressure.pvd")

#  Solving SC below
solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_monitor_true_residual": None,
}
problem_flow = LinearVariationalProblem(a, L, DPP_solution, bcs=bcDPP, constant_jacobian=False)
solver_flow = LinearVariationalSolver(
    problem_flow, options_prefix="dpp_flow", solver_parameters=solver_parameters
)

# Integrating over time
t = dt
step = 1
while t <= T:
    print("============================")
    print("\ttime =", t)
    print("============================")
    c_0.t = t

    solver_flow.solve()

    solve(aAD == LAD, conc, bcs=bcAD)
    conc_k.assign(conc)

    cfile.write(conc, time=t)
    v1file.write(DPP_solution.sub(0), time=t)
    p1file.write(DPP_solution.sub(1), time=t)
    v2file.write(DPP_solution.sub(2), time=t)
    p2file.write(DPP_solution.sub(3), time=t)

    t += dt

print("total time = ", t)
