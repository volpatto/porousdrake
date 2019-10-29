from firedrake import *
import numpy as np
import random
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")

random.seed(222)
nx, ny = 50, 40
Lx, Ly = 5.0, 4.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis("off")

degree = 1
pressure_family = "CG"
velocity_family = "CG"
U = VectorFunctionSpace(mesh, velocity_family, degree)
# U = FunctionSpace(mesh, velocity_family, degree)
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

# k1_bc = interpolate(myk1(), W.sub(0).sub(0))
# k2_bc = interpolate(myk2(), W.sub(2).sub(0))


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

#  Flux BCs
un1_1 = -k1 / mu0
un2_1 = -k2 / mu0
un1_2 = k1 / mu0
un2_2 = k2 / mu0

# un1_1_bc = k1_bc / mu0
# un2_1_bc = k2_bc / mu0
# un1_2_bc = k1_bc / mu0
# un2_2_bc = k2_bc / mu0
#
# bc1 = DirichletBC(W.sub(0).sub(0), un1_1_bc, 1)
# bc2 = DirichletBC(W.sub(0).sub(0), un1_2_bc, 2)
# bc3 = DirichletBC(W.sub(2).sub(0), un2_1_bc, 1)
# bc4 = DirichletBC(W.sub(2).sub(0), un2_2_bc, 2)

# bc1 = DirichletBC(W[0].sub(0), -myk1(), 1)
# bc2 = DirichletBC(W[2].sub(0), -myk2(), 1)
# bc3 = DirichletBC(W[0].sub(0), myk1(), 2)
# bc4 = DirichletBC(W[2].sub(0), myk1(), 2)
# bcs = [bc1, bc2, bc3, bc4]
# bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)

# Source term
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

# Stabilizing parameter
delta_0 = Constant(1.0)
delta_1 = Constant(0.5)
delta_2 = Constant(0.5)
delta_3 = Constant(0.0)

delta_2 = delta_2 * h * h
delta_3 = delta_3 * h * h
eta = Constant(10)

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
a += -delta_2 * alpha1() * (invalpha1() / k1) * (p1 - p2) * div(v1) * dx
a += -delta_2 * alpha2() * (invalpha2() / k2) * (p2 - p1) * div(v2) * dx
###
a += delta_3 * inner(invalpha1() * curl(alpha1() * u1), curl(alpha1() * v1)) * dx
a += delta_3 * inner(invalpha2() * curl(alpha2() * u2), curl(alpha2() * v2)) * dx
# Weakly imposed BC by Nitsche's method
a += dot(v1, n) * p1 * ds + dot(v2, n) * p2 * ds - q1 * dot(u1, n) * ds - q2 * dot(u2, n) * ds
L += -q1 * un1_1 * ds(1) - q2 * un2_1 * ds(1) - q1 * un1_2 * ds(2) - q2 * un2_2 * ds(2)
a += eta / h * inner(dot(v1, n), dot(u1, n)) * ds + eta / h * inner(dot(v2, n), dot(u2, n)) * ds
L += (
    eta / h * dot(v1, n) * un1_1 * ds(1)
    + eta / h * dot(v2, n) * un2_1 * ds(1)
    + eta / h * dot(v1, n) * un1_2 * ds(2)
    + eta / h * dot(v2, n) * un2_2 * ds(2)
)

#  Solving
solver_parameters = {
    "ksp_type": "lgmres",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_rtol": 1e-5,
    "ksp_atol": 1e-5,
    "ksp_monitor_true_residual": None,
}
problem_flow = LinearVariationalProblem(a, L, DPP_solution, bcs=[], constant_jacobian=False)
solver_flow = LinearVariationalSolver(
    problem_flow, options_prefix="dpp_flow", solver_parameters=solver_parameters
)
solver_flow.solve()

# Writing current_solution to .vtk files
# v1file = File('Macro_Velocity_Vpatch.pvd')
# p1file = File('Macro_Pressure_Vpatch.pvd')
# v2file = File('Micro_Velocity_Vpatch.pvd')
# p2file = File('Micro_Pressure_Vpatch.pvd')
# v1file.write(DPP_solution.sub(0))
# p1file.write(DPP_solution.sub(1))
# v2file.write(DPP_solution.sub(2))
# p2file.write(DPP_solution.sub(3))

plot(DPP_solution.sub(0).sub(0))
plt.show()
