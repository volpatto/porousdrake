from firedrake import *
import numpy as np
import random

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

nx, ny = 10, 10
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = UnitSquareMesh(nx, ny, quadrilateral=quadrilateral)

if quadrilateral:
    hdiv_family = "RTCF"
    pressure_family = "DQ"
else:
    hdiv_family = "RT"
    pressure_family = "DG"

plot(mesh)
plt.axis("off")

degree = 1
U = FunctionSpace(mesh, hdiv_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = U * V

n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

f = Function(V)
source_expr = 10 * exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02)
f.interpolate(source_expr)
# f = Constant(0.0)

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

# Boundaries: Left (1), Right (2), Bottom(3), Top (4)
solution = Function(W)
bc1 = DirichletBC(W[0], as_vector([0.0, -sin(5 * x)]), 1)
bc2 = DirichletBC(W[0], as_vector([0.0, sin(5 * y)]), 2)
bcs = [bc1, bc2]

a = (dot(sigma, tau) - div(tau) * u + v * div(sigma)) * dx
a += 0.5 * inner(sigma + grad(u), -tau + grad(v)) * dx
L = f * v * dx + Constant(0.0) * dot(tau, n) * (ds(3) + ds(4))

solve(a == L, solution, bcs=bcs)
sigma_h, u_h = solution.split()
sigma_h.rename("Velocity", "label")
u_h.rename("Pressure", "label")

plot(sigma_h)
plot(u_h)
plt.axis("off")
plt.show()
