from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis("off")

degree = 1
k_plus = 0
primal_family = "DG"
tracer_family = "DGT"
U = FunctionSpace(mesh, primal_family, degree + k_plus)
V = VectorFunctionSpace(mesh, "CG", degree + k_plus)
T = FunctionSpace(mesh, tracer_family, degree)
W = U * T

# Trial and test functions
solution = Function(W)
u, lambda_h = split(solution)
v, mu_h = TestFunction(W)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

# Model parameters
k = Constant(1.0)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))

# Exact solution and source term projection
p_exact = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
sol_exact = Function(U).interpolate(p_exact)
sol_exact.rename("Exact pressure", "label")
sigma_e = Function(V, name="Exact velocity")
sigma_e.project(-(k / mu) * grad(p_exact))
plot(sigma_e)
plt.axis("off")
plt.show()
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(U).interpolate(source_expr)
plot(sol_exact)
plt.axis("off")
plt.show()

# BCs
p_boundaries = Constant(0.0)
vel_projected = sigma_e
bc_multiplier = DirichletBC(W.sub(1), p_boundaries, "on_boundary")

# DG parameter
s = Constant(1.0)
beta = Constant(32.0)
h = CellDiameter(mesh)
h_avg = avg(h)

# Classical term
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
# Hybridization terms
a += s * dot(grad(v), n)("+") * (u("+") - lambda_h("+")) * dS
a += -dot(grad(u), n)("+") * (v("+") - mu_h("+")) * dS
a += (beta / h_avg) * (u("+") - lambda_h("+")) * (v("+") - mu_h("+")) * dS
# Boundary terms
a += (beta / h) * (u - p_boundaries) * v * ds
# L += s * dot(grad(v), n) * p_boundaries * ds

F = a - L

#  Solving SC below
PETSc.Sys.Print("*******************************************\nSolving...\n")
params = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "pmat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    # Use the static condensation PC for hybridized problems
    # and use a direct solve on the reduced system for lambda_h
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0",
    "condensed_field": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

# solve(F == 0, solution)

PETSc.Sys.Print("Solver finished.\n")

# Gathering solution
u_h, lambda_h = solution.split()
u_h.rename("Solution", "label")

# Post-processing solution
sigma_h = Function(V, name="Projected velocity")
sigma_h.project(-(k / mu) * grad(u_h))

output = File("ldgd.pvd", project_output=True)
output.write(u_h, sigma_h)

plot(sigma_h)
plt.axis("off")
plt.show()
plot(u_h)
plt.axis("off")
plt.show()

print("\n*** DoF = %i" % W.dim())
