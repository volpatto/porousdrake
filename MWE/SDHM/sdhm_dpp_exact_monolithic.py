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
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis("off")

degree = 1
pressure_family = "DG"
velocity_family = "DG"
trace_family = "HDiv Trace"
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
T = FunctionSpace(mesh, trace_family, degree)
W = U * V * T * U * V * T

# Trial and test functions
DPP_solution = Function(W)
u1, p1, lambda1, u2, p2, lambda2 = split(DPP_solution)
v1, q1, mu1, v2, q2, mu2 = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)
h = CellDiameter(mesh)

#################################################
# *** Model parameters
#################################################
mu0 = Constant(1.0)
k1 = Constant(1.0)
k2 = Constant(0.1)
b_factor = Constant(1.0)


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

# Exact solution and source term projection
eta = sqrt(b_factor * (k1 + k2) / (k1 * k2))
p_exact_1 = mu0 / pi * exp(pi * x) * sin(pi * y) - mu0 / (b_factor * k1) * exp(eta * y)
p_exact_2 = mu0 / pi * exp(pi * x) * sin(pi * y) + mu0 / (b_factor * k2) * exp(eta * y)
p_e_1 = Function(W.sub(1)).interpolate(p_exact_1)
p_e_1_tr = Function(T).interpolate(p_exact_1)
p_e_1.rename("Exact macro pressure", "label")
p_e_2 = Function(W.sub(4)).interpolate(p_exact_2)
p_e_2_tr = Function(T).interpolate(p_exact_2)
p_e_2.rename("Exact micro pressure", "label")
v_e_1 = Function(W.sub(0), name="Exact macro velocity")
v_e_1.project(-(k1 / mu0) * grad(p_e_1))
v_e_2 = Function(W.sub(3), name="Exact macro velocity")
v_e_2.project(-(k2 / mu0) * grad(p_e_2))

plot(p_e_1)
plot(p_e_2)

bc_multiplier_1 = DirichletBC(W.sub(2), p_e_1_tr, "on_boundary")
bc_multiplier_2 = DirichletBC(W.sub(5), p_e_2_tr, "on_boundary")
bcs = [bc_multiplier_1, bc_multiplier_2]

# Source term
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)

# Stabilizing parameter
beta_0 = Constant(1.0e0)
beta = beta_0 / h
beta_avg = beta_0 / h("+")
delta_0 = Constant(-1.0)
delta_1 = Constant(0.5)
delta_2 = Constant(0.0)
delta_3 = Constant(0.0)

# Mixed classical terms
a = (dot(alpha1() * u1, v1) - div(v1) * p1 - delta_0 * q1 * div(u1)) * dx
a += (dot(alpha2() * u2, v2) - div(v2) * p2 - delta_0 * q2 * div(u2)) * dx
a += delta_0 * q1 * (b_factor * invalpha1() / k1) * (p2 - p1) * dx
a += delta_0 * q2 * (b_factor * invalpha2() / k2) * (p1 - p2) * dx
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
L += delta_2 * alpha1() * (b_factor * invalpha1() / k1) * (p2 - p1) * div(v1) * dx
L += delta_2 * alpha2() * (b_factor * invalpha2() / k2) * (p1 - p2) * div(v2) * dx
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
# Weakly imposed BC from hybridization (Works monolithic, strong BC on traces)
a += beta * invalpha1() * (lambda1 - p_e_1) * (mu1 - q1) * ds
a += beta * invalpha1() * (lambda2 - p_e_2) * (mu2 - q2) * ds
a += (lambda1 * dot(v1, n) + mu1 * (dot(u1, n) - dot(v_e_1, n))) * ds
a += (lambda2 * dot(v2, n) + mu2 * (dot(u2, n) - dot(v_e_2, n))) * ds

F = a - L

#  Monolithic solver
PETSc.Sys.Print("*******************************************\nMonolithic solver.\n")
solver_parameters = {
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_rtol": 1e-5,
    "ksp_max_it": 2000,
    "ksp_monitor_true_residual": True,
}
problem_flow = NonlinearVariationalProblem(F, DPP_solution, bcs=bcs)
solver_flow = NonlinearVariationalSolver(problem_flow, solver_parameters=solver_parameters)
solver_flow.solve()

# Writing solution in a .vtk file and plotting the solution
plot(DPP_solution.sub(1))
plot(DPP_solution.sub(4))
plt.show()

output_file = File("dpp_sdhm_exact_mono.pvd")
v1_sol = DPP_solution.sub(0)
v1_sol.rename("Macro velocity", "label")
p1_sol = DPP_solution.sub(1)
p1_sol.rename("Macro pressure", "label")
v2_sol = DPP_solution.sub(3)
v2_sol.rename("Micro velocity", "label")
p2_sol = DPP_solution.sub(4)
p2_sol.rename("Micro pressure", "label")
output_file.write(p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2)
