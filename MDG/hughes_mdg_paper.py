from firedrake import *
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

if quadrilateral:
    pressure_family = 'DQ'
else:
    pressure_family = 'DG'

plot(mesh)
plt.axis('off')

degree = 1
U = VectorFunctionSpace(mesh, "DG", degree + 1)
V = FunctionSpace(mesh, pressure_family, degree)
W = MixedFunctionSpace([U, V])

v, p = TrialFunctions(W)
w, q = TestFunctions(W)
solution = Function(W)

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

# Model parameters
k = Constant(1.0)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))

# Boundaries: Left (1), Right (2), Bottom(3), Top (4)
vx = -2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
vy = -2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly)
p_boundaries = Constant(0.0)
v_projected = sigma_e
# An alternative for the BC
# v_projected = project(as_vector([vx, vy]), W[0])

# Stabilizing parameters
h = CellDiameter(mesh)
h_avg = (h('+') + h('-')) / 2.
beta = Constant(0.0)

# Regular DG terms
a = dot(w, (mu / k) * v) * dx - \
    div(w) * p * dx + \
    q * div(v) * dx + \
    jump(w, n) * avg(p) * dS - \
    avg(q) * jump(v, n) * dS + \
    (dot(w, n) * p - dot(v, n) * q) * (ds(1) + ds(2) + ds(3) + ds(4))
L = f * q * dx - dot(rho * g, w) * dx - p_boundaries * dot(w, n) * (ds(1) + ds(2) + ds(3) + ds(4)) - \
    dot(v_projected, n) * q * (ds(1) + ds(2)) - dot(v_projected, n) * q * (ds(3) + ds(4))
# Stabilizing terms
a += 0.5 * dot(-(mu / k) * w + grad(q), (k / mu) * ((mu / k) * v + grad(p))) * dx + \
    (beta / h_avg) * avg(k / mu) * dot(jump(q, n), jump(p, n)) * dS
L += 0.5 * dot((k / mu) * rho * g, - (mu / k) * w + grad(q)) * dx

solver_parameters = {
    # 'ksp_type': 'tfqmr',
    'ksp_type': 'gmres',
    'pc_type': 'bjacobi',
    'mat_type': 'aij',
    'ksp_rtol': 1e-3,
    'ksp_max_it': 2000,
    'ksp_monitor': False
}

solve(a == L, solution, solver_parameters=solver_parameters)
sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

output = File('hughes_paper_MDG.pvd', project_output=True)
output.write(sigma_h, u_h, sol_exact, sigma_e)

plot(sigma_h)
plot(u_h)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % W.dim())
