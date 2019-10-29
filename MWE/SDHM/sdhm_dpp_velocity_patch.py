from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")

nx, ny = 10, 10
Lx, Ly = 5.0, 4.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis("off")

degree = 1
pressure_family = "DG"
velocity_family = "DG"
trace_family = "HDiv Trace"
U = VectorFunctionSpace(mesh, velocity_family, degree + 1)
V = FunctionSpace(mesh, pressure_family, degree)
T = FunctionSpace(mesh, trace_family, degree)
W = U * V * T * U * V * T

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
kSpace = FunctionSpace(mesh, "DG", 0)

mu0 = Constant(1.0)
k = Constant(0.2)
tol = 1e-14


class myk1(Expression):
    def eval(self, values, x):
        if x[1] <= 0.8:
            values[0] = 80 * k
        elif x[1] <= 1.6:
            values[0] = 30 * k
        elif x[1] <= 2.4:
            values[0] = 5 * k
        elif x[1] <= 3.2:
            values[0] = 50 * k
        elif x[1] <= 4.0:
            values[0] = 10 * k


class myk2(Expression):
    def eval(self, values, x):
        if x[1] <= 0.8:
            values[0] = 16 * k
        elif x[1] <= 1.6:
            values[0] = 6 * k
        elif x[1] <= 2.4:
            values[0] = 1 * k
        elif x[1] <= 3.2:
            values[0] = 10 * k
        elif x[1] <= 4.0:
            values[0] = 2 * k


k1 = interpolate(myk1(), kSpace)
k2 = interpolate(myk2(), kSpace)


def alpha1():
    return mu0 / k1


def invalpha1():
    return 1.0 / alpha1()


def alpha2():
    return mu0 / k2


def invalpha2():
    return 1.0 / alpha2()


#################################################
#################################################
#################################################

# Flux BCs
un1_1 = -k1 / mu0
un2_1 = -k2 / mu0
un1_2 = k1 / mu0
un2_2 = k2 / mu0

# Source term
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

# Stabilizing parameter
beta_0 = Constant(1.0)
beta = beta_0 / h
beta_avg = beta_0 / h("+")
delta_0 = Constant(1.0)
delta_1 = Constant(-0.5)
delta_2 = Constant(0.5)
delta_3 = Constant(0.5)

# Mixed classical terms
a = (dot(alpha1() * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
a += (dot(alpha2() * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
a += delta_0 * q1 * (invalpha1() / k1) * (p1 - p2) * dx
a += delta_0 * q2 * (invalpha2() / k2) * (p2 - p1) * dx
L = -delta_0 * dot(rhob1, v1) * dx
L += -delta_0 * dot(rhob2, v2) * dx
# Stabilizing terms
###
a += (
    delta_1
    * inner(invalpha1() * (alpha1() * u1 + grad(p1)), delta_0 * alpha1() * v1 + grad(q1))
    * dx
)
a += (
    delta_1
    * inner(invalpha2() * (alpha2() * u2 + grad(p2)), delta_0 * alpha2() * v2 + grad(q2))
    * dx
)
###
a += delta_2 * alpha1() * div(u1) * div(v1) * dx
a += delta_2 * alpha2() * div(u2) * div(v2) * dx
L += delta_2 * alpha1() * (invalpha1() / k1) * (p1 - p2) * div(v1) * dx
L += delta_2 * alpha2() * (invalpha2() / k2) * (p2 - p1) * div(v2) * dx
###
a += delta_3 * inner(invalpha1() * curl(alpha1() * u1), curl(alpha1() * v1)) * dx
a += delta_3 * inner(invalpha2() * curl(alpha2() * u2), curl(alpha2() * v2)) * dx
# Hybridization terms
###
a += lambda1("+") * jump(v1, n) * dS + mu1("+") * jump(u1, n) * dS
a += lambda2("+") * jump(v2, n) * dS + mu2("+") * jump(u2, n) * dS
###
a += beta_avg * invalpha1()("+") * (lambda1("+") - p1("+")) * (mu1("+") - q1("+")) * dS
a += beta_avg * invalpha2()("+") * (lambda2("+") - p2("+")) * (mu2("+") - q2("+")) * dS
# Weakly imposed BC from hybridization
a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n) - un1_1)) * ds(1)
a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n) - un2_1)) * ds(1)
a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n) - un1_2)) * ds(2)
a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n) - un2_2)) * ds(2)
a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n))) * (ds(3) + ds(4))
a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n))) * (ds(3) + ds(4))

F = a - L

#  Solving SC below
PETSc.Sys.Print("*******************************************\nSolving using static condensation.\n")
solver_parameters = {
    "snes_type": "ksponly",
    "pmat_type": "matfree",
    # 'ksp_view': True,
    "ksp_type": "gmres",
    "ksp_monitor_true_residual": True,
    # 'snes_monitor': True,
    "ksp_rtol": 1.0e-5,
    "ksp_atol": 1.0e-5,
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
solver_parameters = {
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_rtol": 1e-5,
    "ksp_max_it": 2000,
    "ksp_monitor_true_residual": True,
}
problem_flow = NonlinearVariationalProblem(F, DPP_solution, bcs=[])
solver_flow = NonlinearVariationalSolver(problem_flow, solver_parameters=solver_parameters)
solver_flow.solve()

# Writing .vtk solution output
v1file = File("Macro_Velocity_Vpatch.pvd")
p1file = File("Macro_Pressure_Vpatch.pvd")
v2file = File("Micro_Velocity_Vpatch.pvd")
p2file = File("Micro_Pressure_Vpatch.pvd")
v1file.write(DPP_solution.sub(0))
p1file.write(DPP_solution.sub(1))
v2file.write(DPP_solution.sub(3))
p2file.write(DPP_solution.sub(4))
