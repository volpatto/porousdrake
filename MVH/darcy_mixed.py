from firedrake import *
import numpy as np
import random
try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

nx, ny = 10, 10
Lx, Ly = 1.0, 1.0
quadrilateral = False

if quadrilateral:
    DG = 'DQ'
    hdiv_family = 'RTCF'
else:
    DG = 'DG'
    hdiv_family = 'RT'

mesh = UnitSquareMesh(nx, ny, quadrilateral=quadrilateral)

plot(mesh)
plt.axis('off')

degree = 1
U = FunctionSpace(mesh, hdiv_family, degree)
V = FunctionSpace(mesh, DG, degree - 1)
W = U * V

n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

f = Function(V).interpolate(10 * exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

solution = Function(W)
bc1 = DirichletBC(W[0], as_vector([0.0, -sin(5*x)]), 1)
bc2 = DirichletBC(W[0], as_vector([0.0, sin(5*y)]), 2)
bcs = [bc1, bc2]

a = (dot(sigma, tau) - div(tau) * u + v * div(sigma)) * dx
L = f * v * dx + Constant(0.0) * dot(tau, n) * (ds(3) + ds(4))

solve(a == L, solution, bcs=bcs)
sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

plot(sigma_h)
plot(u_h)
plt.show()
