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
primal_family = 'DG'
U = FunctionSpace(mesh, primal_family, degree)
V = VectorFunctionSpace(mesh, primal_family, degree)

# Trial and test functions
u = TrialFunction(U)
v = TestFunction(U)

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
sol_exact.rename('Exact pressure', 'label')
sigma_e = Function(V, name='Exact velocity')
sigma_e.project(-(k / mu) * grad(p_exact))
plot(sigma_e)
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(U).interpolate(source_expr)
plot(sol_exact)
plt.axis('off')

# Boundaries value
p_boundaries = Constant(0.0)

# DG parameter
s = Constant(-1.0)
beta = Constant(20.0)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-')) / 2.

# Classical term
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
# DG terms
a += s * (dot(jump(u, n), avg(grad(v))) - dot(jump(v, n), avg(grad(u)))) * dS
a += (beta / h_avg) * dot(jump(u, n), jump(v, n)) * dS
a += (beta / h) * inner(u, v) * ds
# DG boundary condition terms
L += s * dot(grad(v), n) * p_boundaries * ds \
     + (beta / h) * p_boundaries * v * ds \
     + v * dot(sigma_e, n) * ds

#  Solving SC below
PETSc.Sys.Print("*******************************************\nSolving...\n")
solver_parameters = {
    'ksp_type': 'gmres',
    'pc_type': 'bjacobi',
    'mat_type': 'aij',
    'ksp_rtol': 1e-3,
    'ksp_max_it': 2000,
    'ksp_monitor': False
}

solution = Function(U)
problem = LinearVariationalProblem(a, L, solution)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

PETSc.Sys.Print("Solver finished.\n")

# Post-processing solution
sigma_h = Function(V, name='Projected velocity')
sigma_h.project(-(k / mu) * grad(solution))

output = File('dg.pvd', project_output=True)
output.write(solution, sigma_h)

plot(sigma_h)
plot(solution)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % U.dim())
