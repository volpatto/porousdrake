from firedrake import *
from scipy import sparse
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
from porousdrake.post_processing.plot_sparsity import (
    plot_matrix_hybrid_multiplier_spp,
    plot_matrix_hybrid_full,
)

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")

nx, ny = 4, 4
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
W = U * V * T

# Trial and test functions
solution = Function(W)
u, p, lambda_h = TrialFunctions(W)
v, q, mu_h = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)
h = CellDiameter(mesh)

# Model parameters
k = Constant(1.0)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))

# Exact solution and source term projection
p_exact = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
sol_exact = Function(V).interpolate(p_exact)
sol_exact.rename("Exact pressure", "label")
sigma_e = Function(U, name="Exact velocity")
sigma_e.project(-(k / mu) * grad(p_exact))
plot(sigma_e)
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(V).interpolate(source_expr)
plot(sol_exact)
plt.axis("off")

# BCs
p_boundaries = Constant(0.0)
v_projected = sigma_e
bc_multiplier = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

# Hybridization parameter
beta_0 = Constant(1.0)
beta = beta_0 / h
beta_avg = beta_0 / h("+")

# Mixed classical terms
a = (dot((mu / k) * u, v) - div(v) * p - q * div(u)) * dx
L = -f * q * dx - dot(rho * g, v) * dx
# Stabilizing terms
a += -0.5 * inner((k / mu) * ((mu / k) * u + grad(p)), (mu / k) * v + grad(q)) * dx
a += 0.5 * (mu / k) * div(u) * div(v) * dx
a += 0.5 * inner((k / mu) * curl((mu / k) * u), curl((mu / k) * v)) * dx
L += 0.5 * (mu / k) * f * div(v) * dx
# Hybridization terms
a += lambda_h("+") * dot(v, n)("+") * dS + mu_h("+") * dot(u, n)("+") * dS
a += beta_avg * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS

F = a - L

plot_matrix_hybrid_multiplier_spp(a, bcs=bc_multiplier)
plt.show()

plot_matrix_hybrid_full(a, bcs=bc_multiplier)
plt.show()

print("\n*** DoF = %i" % W.sub(2).dim())
