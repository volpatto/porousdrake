from firedrake import *

# kSpace = FunctionSpace(mesh, "DG", 0)

mu = Constant(1.0)
k_ref = Constant(0.2)
rhob = Constant((0.0, 0.0))
f = Constant(0.0)
tol = 1e-14


class myk(Expression):
    def eval(self, values, x):
        if x[1] <= 0.8:
            values[0] = 80 * k_ref
        elif x[1] <= 1.6:
            values[0] = 30 * k_ref
        elif x[1] <= 2.4:
            values[0] = 5 * k_ref
        elif x[1] <= 3.2:
            values[0] = 50 * k_ref
        elif x[1] <= 4.0:
            values[0] = 10 * k_ref
