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
pressure_family = "DG"
velocity_family = "DG"
trace_family = "HDiv Trace"
U = VectorFunctionSpace(mesh, velocity_family, degree)
U1 = VectorFunctionSpace(mesh, velocity_family, degree)
U2 = VectorFunctionSpace(mesh, velocity_family, degree + 1)
V = FunctionSpace(mesh, pressure_family, degree)
T = FunctionSpace(mesh, trace_family, degree)
W = U1 * V * T * U2 * V * T

# Trial and test functions
DPP_solution = Function(W)
u1, p1, lambda1, u2, p2, lambda2 = split(DPP_solution)
v1, q1, mu1, v2, q2, mu2 = TestFunctions(W)

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


def alpha1_avg(c):
    return mu0 * exp(Rc * (1.0 - c)) / k1("+")


def invalpha1(c):
    return 1.0 / alpha1(c)


def invalpha1_avg(c):
    return 1.0 / alpha1_avg(c)


def alpha2(c):
    return mu0 * exp(Rc * (1.0 - c)) / k2


def alpha2_avg(c):
    return mu0 * exp(Rc * (1.0 - c)) / k2("+")


def invalpha2(c):
    return 1.0 / alpha2(c)


def invalpha2_avg(c):
    return 1.0 / alpha2_avg(c)


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
v_topbottom = Constant(0.0)
p_L = Constant(10.0)
p_R = Constant(1.0)
bcDPP = []
c_inj = Constant(1.0)
bcleft_c = DirichletBC(uSpace, c_inj, 1)
bcAD = [bcleft_c]

# Time parameters
T = 1.5e-3
dt = 5e-5

# Source term
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

# Stabilizing parameter
beta_0 = Constant(1.0e-15)
beta = beta_0 / h
beta_avg = beta_0 / h("+")
# beta = Constant(0.0)
delta_0 = Constant(1.0)
delta_1 = Constant(-0.5)
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
L += delta_2 * alpha1(conc_k) * (invalpha1(conc_k) / k1) * (p2 - p1) * div(v1) * dx
L += delta_2 * alpha2(conc_k) * (invalpha2(conc_k) / k2) * (p1 - p2) * div(v2) * dx
###
a += delta_3 * inner(invalpha1(conc_k) * curl(alpha1(conc_k) * u1), curl(alpha1(conc_k) * v1)) * dx
a += delta_3 * inner(invalpha2(conc_k) * curl(alpha2(conc_k) * u2), curl(alpha2(conc_k) * v2)) * dx
# Hybridization terms
###
a += lambda1("+") * jump(v1, n) * dS + mu1("+") * jump(u1, n) * dS
a += lambda2("+") * jump(v2, n) * dS + mu2("+") * jump(u2, n) * dS
###
a += beta_avg * invalpha1_avg(conc_k("+")) * (lambda1("+") - p1("+")) * (mu1("+") - q1("+")) * dS
a += beta_avg * invalpha2_avg(conc_k("+")) * (lambda2("+") - p2("+")) * (mu2("+") - q2("+")) * dS
# a += beta_avg * (lambda1('+') - p1('+')) * (mu1('+') - q1('+')) * dS
# a += beta_avg * (lambda2('+') - p2('+')) * (mu2('+') - q2('+')) * dS
# Weakly imposed BC from hybridization
a += (p_L * dot(v1, n) + mu1 * dot(u1, n)) * ds(1)
a += (p_L * dot(v2, n) + mu2 * dot(u2, n)) * ds(1)
a += (p_R * dot(v1, n) + mu1 * dot(u1, n)) * ds(2)
a += (p_R * dot(v2, n) + mu2 * dot(u2, n)) * ds(2)
a += (lambda1 * dot(v1, n) + mu1 * dot(u1, n)) * (ds(3) + ds(4))
a += (lambda2 * dot(v2, n) + mu2 * dot(u2, n)) * (ds(3) + ds(4))
###
# a += beta * invalpha1(conc_k) * (lambda1 - p1) * (mu1 - q1) * (ds(3) + ds(4))
# a += beta * invalpha2(conc_k) * (lambda2 - p2) * (mu2 - q2) * (ds(3) + ds(4))
a += beta * invalpha1(conc_k) * lambda1 * mu1 * (ds(3) + ds(4))
a += beta * invalpha2(conc_k) * lambda2 * mu2 * (ds(3) + ds(4))
a += beta * invalpha1(conc_k) * (lambda1 - p_L) * mu1 * ds(1)
a += beta * invalpha2(conc_k) * (lambda2 - p_L) * mu2 * ds(1)
a += beta * invalpha1(conc_k) * (lambda1 - p_R) * mu1 * ds(2)
a += beta * invalpha2(conc_k) * (lambda2 - p_R) * mu2 * ds(2)

F = a - L

# *** Transport problem
vnorm = sqrt(
    dot((DPP_solution.sub(0) + DPP_solution.sub(3)), (DPP_solution.sub(0) + DPP_solution.sub(3)))
)

taw = h / (2.0 * vnorm) * dot((DPP_solution.sub(0) + DPP_solution.sub(3)), grad(u))

a_r = (
    taw
    * (c1 + dt * (dot((DPP_solution.sub(0) + DPP_solution.sub(3)), grad(c1)) - div(D * grad(c1))))
    * dx
)

L_r = taw * (conc_k + dt * f) * dx

aAD = (
    a_r
    + u * c1 * dx
    + dt
    * (
        u * dot((DPP_solution.sub(0) + DPP_solution.sub(3)), grad(c1)) * dx
        + dot(grad(u), D * grad(c1)) * dx
    )
)
LAD = L_r + u * conc_k * dx + dt * u * f * dx

save_path = "results2_%s/fine_micro" % os.path.basename(__file__)[:-3]
os.makedirs(save_path, exist_ok=True)

cfile = File(save_path + "/Concentration.pvd")
v1file = File(save_path + "/Macro_Velocity.pvd")
p1file = File(save_path + "/Macro_Pressure.pvd")
v2file = File(save_path + "/Micro_Velocity.pvd")
p2file = File(save_path + "/Micro_Pressure.pvd")

#  Solving SC below
PETSc.Sys.Print("*******************************************\nSolving using static condensation.\n")
solver_parameters = {
    "snes_type": "ksponly",
    "pmat_type": "matfree",
    # 'ksp_view': True,
    "ksp_type": "lgmres",
    "ksp_monitor_true_residual": None,
    # 'snes_monitor': True,
    "ksp_rtol": 1.0e-10,
    "ksp_atol": 1.0e-10,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_0_fields": "0,1,2",
    "pc_fieldsplit_1_fields": "3,4,5",
    "fieldsplit_0": {
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    },
    "fieldsplit_1": {
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    },
}
problem_flow = NonlinearVariationalProblem(F, DPP_solution, bcs=bcDPP)
solver_flow = NonlinearVariationalSolver(problem_flow, solver_parameters=solver_parameters)

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
    v2file.write(DPP_solution.sub(3), time=t)
    p2file.write(DPP_solution.sub(4), time=t)

    t += dt

print("total time = ", t)
