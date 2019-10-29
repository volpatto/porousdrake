"""
The present code is based in the code and method proposed in [1], but adapted from [2]. Here I just use as a reference
code, since the mentioned paper is a benchmark for my own work.

Ref:
[1] S.H.S. Joodat, K.B. Nakshatrala, R. Ballarini, Modeling flow in porous media with double porosity/permeability:
A stabilized mixed formulation, error analysis, and numerical solutions, Computer Methods in Applied Mechanics and
Engineering, Volume 337, 2018, Pages 632-676, ISSN 0045-7825, https://doi.org/10.1016/j.cma.2018.04.004.
[2] Joshaghani, M. S., S. H. S. Joodat, and K. B. Nakshatrala. "A stabilized mixed discontinuous Galerkin formulation
for double porosity/permeability model." arXiv preprint arXiv:1805.01389 (2018).
"""
from firedrake import *
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

nx, ny = 50, 40
Lx, Ly = 5.0, 4.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

degree = 1
velSpace = VectorFunctionSpace(mesh, "CG", degree)
pSpace = FunctionSpace(mesh, "CG", degree)
wSpace = MixedFunctionSpace([velSpace, pSpace, velSpace, pSpace])

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


un1_1 = -k1 / mu0
un2_1 = -k2 / mu0
un1_2 = k1 / mu0
un2_2 = k2 / mu0

(v1, p1, v2, p2) = TrialFunctions(wSpace)
(w1, q1, w2, q2) = TestFunctions(wSpace)
DPP_solution = Function(wSpace)

rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
eta = Constant(10)

aDPP = (
    dot(w1, alpha1() * v1) * dx
    + dot(w2, alpha2() * v2) * dx
    - div(w1) * p1 * dx
    - div(w2) * p2 * dx
    + q1 * div(v1) * dx
    + q2 * div(v2) * dx
    + q1 * (invalpha1() / k1) * (p1 - p2) * dx
    - q2 * (invalpha2() / k2) * (p1 - p2) * dx
    - 0.5 * dot(alpha1() * w1 - grad(q1), invalpha1() * (alpha1() * v1 + grad(p1))) * dx
    - 0.5 * dot(alpha2() * w2 - grad(q2), invalpha2() * (alpha2() * v2 + grad(p2))) * dx
)
LDPP = (
    dot(w1, rhob1) * dx
    + dot(w2, rhob2) * dx
    - q1 * un1_1 * ds(1)
    - q2 * un2_1 * ds(1)
    - q1 * un1_2 * ds(2)
    - q2 * un2_2 * ds(2)
    - 0.5 * dot(alpha1() * w1 - grad(q1), invalpha1() * rhob1) * dx
    - 0.5 * dot(alpha2() * w2 - grad(q2), invalpha2() * rhob2) * dx
)
# Nitsche's method
aDPP += dot(w1, n) * p1 * ds + dot(w2, n) * p2 * ds - q1 * dot(v1, n) * ds - q2 * dot(v2, n) * ds
aDPP += eta / h * inner(dot(w1, n), dot(v1, n)) * ds + eta / h * inner(dot(w2, n), dot(v2, n)) * ds
LDPP += (
    eta / h * dot(w1, n) * un1_1 * ds(1)
    + eta / h * dot(w2, n) * un2_1 * ds(1)
    + eta / h * dot(w1, n) * un1_2 * ds(2)
    + eta / h * dot(w2, n) * un2_2 * ds(2)
)

solver_parameters = {
    "ksp_type": "lgmres",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_rtol": 1e-5,
    "ksp_monitor_true_residual": True,
}

problem_flow = LinearVariationalProblem(aDPP, LDPP, DPP_solution, bcs=[], constant_jacobian=False)
solver_flow = LinearVariationalSolver(
    problem_flow, options_prefix="flow_", solver_parameters=solver_parameters
)
solver_flow.solve()

v1file = File("Macro_Velocity_Vpatch.pvd")
p1file = File("Macro_Pressure_Vpatch.pvd")
v2file = File("Micro_Velocity_Vpatch.pvd")
p2file = File("Micro_Pressure_Vpatch.pvd")

v1file.write(DPP_solution.sub(0))
p1file.write(DPP_solution.sub(1))
v2file.write(DPP_solution.sub(2))
p2file.write(DPP_solution.sub(3))

plot(DPP_solution.sub(0).sub(0))
plt.show()
