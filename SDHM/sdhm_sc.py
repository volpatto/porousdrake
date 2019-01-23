from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis('off')

degree = 1
pressure_family = 'DG'
velocity_family = 'DG'
trace_family = 'HDiv Trace'
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
T = FunctionSpace(mesh, trace_family, degree)
W = U * V * T

# Trial and test functions
solution = Function(W)
u, p, lambda_h = split(solution)
v, q, mu_h = TestFunctions(W)

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
sol_exact = Function(V).interpolate(p_exact)
sol_exact.rename('Exact pressure', 'label')
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-(k / mu) * grad(p_exact))
plot(sigma_e)
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(V).interpolate(source_expr)
plot(sol_exact)
plt.axis('off')

# BCs
p_boundaries = Constant(0.0)
v_projected = sigma_e
bc_multiplier = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

# Hybridization parameter
beta = Constant(0.0)

# Mixed classical terms
a = (dot((mu / k) * u, v) - div(v) * p - q * div(u)) * dx
L = -f * q * dx - dot(rho * g, v) * dx - p_boundaries * dot(v, n) * (ds(1) + ds(2) + ds(3) + ds(4))
# Stabilizing terms
a += -0.5 * inner((k / mu) * ((mu / k) * u + grad(p)), (mu / k) * v + grad(q)) * dx
a += 0.5 * (mu / k) * div(u) * div(v) * dx
a += 0.5 * inner((k / mu) * curl((mu / k) * u), curl((mu / k) * v)) * dx
L += 0.5 * (mu / k) * f * div(v) * dx
# Hybridization terms
a += lambda_h('+') * dot(v, n)('+') * dS + mu_h('+') * dot(u, n)('+') * dS
a += beta * (lambda_h('+') - p('+')) * (mu_h('+') - q('+')) * dS
# Weakly imposed BC
a += (lambda_h * dot(v, n) + mu_h * dot(u, n)) * (ds(1) + ds(2) + ds(3) + ds(4))
a += beta * (lambda_h - p) * (mu_h - q) * (ds(1) + ds(2) + ds(3) + ds(4))
L += (lambda_h * dot(v, n) + mu_h * dot(v_projected, n)) * (ds(1) + ds(2) + ds(3) + ds(4))
L += beta * (lambda_h - p_boundaries) * (mu_h - q) * (ds(1) + ds(2) + ds(3) + ds(4))

F = a - L

#  Solving SC below
PETSc.Sys.Print("*******************************************\nSolving using static condensation.\n")
params = {'snes_type': 'ksponly',
          'mat_type': 'matfree',
          'pmat_type': 'matfree',
          'ksp_type': 'preonly',
          'pc_type': 'python',
          # Use the static condensation PC for hybridized problems
          # and use a direct solve on the reduced system for lambda_h
          'pc_python_type': 'firedrake.SCPC',
          'pc_sc_eliminate_fields': '0, 1',
          'condensed_field': {'ksp_type': 'preonly',
                        'pc_type': 'lu',
                        'pc_factor_mat_solver_type': 'mumps'}}

problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

PETSc.Sys.Print("Solver finished.\n")

# Gathering the results
sigma_h, u_h, lamb = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

output = File('sdhm_sc.pvd', project_output=True)
output.write(sigma_h, u_h, sol_exact, sigma_e)

plot(sigma_h)
plot(u_h)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % W.dim())
