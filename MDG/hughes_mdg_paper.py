from firedrake import *
import numpy as np
import random
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
except:
    warning("Matplotlib not imported")

nx, ny = 100, 100
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = UnitSquareMesh(nx, ny, quadrilateral=quadrilateral)

if quadrilateral:
    pressure_family = 'DQ'
else:
    pressure_family = 'DG'

plot(mesh)
plt.axis('off')

degree = 2
U = VectorFunctionSpace(mesh, "DG", degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = MixedFunctionSpace([U, V])

n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

p_exact = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
sol_exact = Function(V).interpolate(p_exact)
sol_exact.rename('Exact pressure', 'label')
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))
plot(sigma_e)
source_expr = div(-grad(p_exact))
f = Function(V).interpolate(source_expr)
plot(sol_exact)
plt.axis('off')

v, p = TrialFunctions(wSpace)
w, q = TestFunctions(wSpace)
solution = Function(wSpace)

# Boundaries: Left (1), Right (2), Bottom(3), Top (4)
vx = -2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
vy = -2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly)
p_boundaries = Constant(0.0)
bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
bcs = [bc1, bc2, bc3, bc4]

#########################################################
#########################################################
#########################################################

h = CellSize(mesh)
h_avg = (h('+') + h('-')) / 2.

eta_p = Constant(0.0)

a = dot(w, alpha(conc_k) * v) * dx - \
    div(w) * p * dx + \
    q * div(v) * dx + \
    jump(w, n) * avg(p) * dS - \
    avg(q) * jump(v, n) * dS + \
    (dot(w, n) * p - dot(v, n) * q) * (ds(3) + ds(4)) + \
    0.5 * dot(alpha(conc_k) * w - grad(q), \
             invalpha(conc_k) * (alpha(conc_k) * v + grad(p))) * dx + \
    (eta_p / h_avg) * avg(1. / alpha(conc_k)) * dot(jump(q, n), jump(p, n)) * dS

#########################################################
#########################################################
#########################################################

a = (dot(sigma, tau) - div(tau) * u + v * div(sigma)) * dx
a += 0.5 * inner(sigma + grad(u), - tau + grad(v)) * dx
L = f * v * dx - Constant(0.0) * dot(tau, n) * (ds(1) + ds(2) + ds(3) + ds(4))

solver_parameters = {
    #'ksp_type': 'tfqmr',
    'ksp_type': 'gmres',
    'pc_type': 'bjacobi',
    'mat_type': 'aij',
    'ksp_rtol': 1e-3,
    'ksp_max_it': 2000,
    'ksp_monitor': True
}

solve(a == L, solution, bcs=bcs, solver_parameters=solver_parameters)
sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

output = File('hughes_paper.pvd', project_output=True)
output.write(sigma_h, u_h, sol_exact, sigma_e)

plot(sigma_h)
plot(u_h)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % W.dim())
