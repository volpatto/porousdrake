"""
The present code was originally provided by the paper [1]. Here I just use as a reference code, since the mentioned
paper is a benchmark for my own work.

Ref:
[1] Joshaghani, M. S., S. H. S. Joodat, and K. B. Nakshatrala. "A stabilized mixed discontinuous Galerkin formulation
for double porosity/permeability model." arXiv preprint arXiv:1805.01389 (2018).
"""
from firedrake import *
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)
h = CellDiameter(mesh)

degree = 1
velSpace = VectorFunctionSpace(mesh, "DG", degree)
pSpace = FunctionSpace(mesh, "DG", degree)
wSpace = MixedFunctionSpace([velSpace, pSpace, velSpace, pSpace])

mu0 = Constant(1.0)
k1 = Constant(1.0)
k2 = Constant(0.1)
b_factor = Constant(1.0)


def alpha1():
    return mu0 / k1


def invalpha1():
    return 1. / alpha1()


def alpha2():
    return mu0 / k2


def invalpha2():
    return 1. / alpha2()


# Exact solution and source term projection
eta = sqrt(b_factor * (k1 + k2) / (k1 * k2))
p_exact_1 = mu0 / pi * exp(pi * x) * sin(pi * y) - mu0 / (b_factor * k1) * exp(eta * y)
p_exact_2 = mu0 / pi * exp(pi * x) * sin(pi * y) + mu0 / (b_factor * k2) * exp(eta * y)
p_e_1 = Function(pSpace).interpolate(p_exact_1)
p_e_1.rename('Exact macro pressure', 'label')
p_e_2 = Function(pSpace).interpolate(p_exact_2)
p_e_2.rename('Exact micro pressure', 'label')
v_e_1 = Function(velSpace, name='Exact macro velocity')
v_e_1.project(-(k1 / mu0) * grad(p_e_1))
v_e_2 = Function(velSpace, name='Exact macro velocity')
v_e_2.project(-(k2 / mu0) * grad(p_e_2))

plot(v_e_1)
plot(v_e_2)

(v1, p1, v2, p2) = TrialFunctions(wSpace)
(w1, q1, w2, q2) = TestFunctions(wSpace)
DPP_solution = Function(wSpace)

rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

h_avg = (h('+') + h('-')) / 2.
eta_p, eta_u = Constant(0.0), Constant(0.0)

aDPP = dot(w1, alpha1() * v1) * dx + \
    dot(w2, alpha2() * v2) * dx - \
    div(w1) * p1 * dx - \
    div(w2) * p2 * dx + \
    q1 * div(v1) * dx + \
    q2 * div(v2) * dx + \
    q1 * (b_factor * invalpha1() / k1) * (p1 - p2) * dx - \
    q2 * (b_factor * invalpha2() / k2) * (p1 - p2) * dx + \
    jump(w1, n) * avg(p1) * dS + \
    jump(w2, n) * avg(p2) * dS - \
    avg(q1) * jump(v1, n) * dS - \
    avg(q2) * jump(v2, n) * dS - \
    0.5 * dot(alpha1() * w1 - grad(q1), invalpha1() * (alpha1() * v1 + grad(p1))) * dx - \
    0.5 * dot(alpha2() * w2 - grad(q2), invalpha2() * (alpha2() * v2 + grad(p2))) * dx + \
    (eta_u * h_avg) * avg(alpha1()) * (jump(v1, n) * jump(w1, n)) * dS + \
    (eta_u * h_avg) * avg(alpha2()) * (jump(v2, n) * jump(w2, n)) * dS + \
    (eta_p / h_avg) * avg(1. / alpha1()) * dot(jump(q1, n), jump(p1, n)) * dS + \
    (eta_p / h_avg) * avg(1. / alpha2()) * dot(jump(q2, n), jump(p2, n)) * dS
LDPP = dot(w1, rhob1) * dx + \
    dot(w2, rhob2) * dx - \
    dot(w1, n) * p_e_1 * ds - \
    dot(w2, n) * p_e_2 * ds - \
    0.5 * dot(alpha1() * w1 - grad(q1), invalpha1() * rhob1) * dx - \
    0.5 * dot(alpha2() * w2 - grad(q2), invalpha2() * rhob2) * dx

solver_parameters = {
    'ksp_type': 'lgmres',
    'pc_type': 'lu',
    'mat_type': 'aij',
    'ksp_rtol': 1e-5,
    'ksp_monitor_true_residual': True
}

problem_flow = LinearVariationalProblem(aDPP, LDPP, DPP_solution, bcs=[], constant_jacobian=False)
solver_flow = LinearVariationalSolver(problem_flow, options_prefix='flow_', solver_parameters=solver_parameters)
solver_flow.solve()

plot(DPP_solution.sub(0))
plot(DPP_solution.sub(2))
plt.show()

output_file = File('results_dg-jh/dpp_exact_mdg.pvd')
v1_sol = DPP_solution.sub(0)
v1_sol.rename('Macro velocity', 'label')
p1_sol = DPP_solution.sub(1)
p1_sol.rename('Macro pressure', 'label')
v2_sol = DPP_solution.sub(2)
v2_sol.rename('Micro velocity', 'label')
p2_sol = DPP_solution.sub(3)
p2_sol.rename('Micro pressure', 'label')
output_file.write(p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2)
