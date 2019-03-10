from firedrake import *

# kSpace = FunctionSpace(mesh, "DG", 0)

mu0 = Constant(1.0)
k = Constant(0.2)
b_factor = Constant(1)
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
tol = 1e-14


class myk1(Expression):
    def eval(self, values, x):
        if x[1] <= 0.8:
            values[0] = 80 * k
        elif x[1] <= 1.6:
            values[0] = 30 * k
        elif x[1] <= 2.4:
            values[0] = 5 * k
        elif x[1] <= 3.2:
            values[0] = 50 * k
        elif x[1] <= 4.0:
            values[0] = 10 * k


class myk2(Expression):
    def eval(self, values, x):
        if x[1] <= 0.8:
            values[0] = 16 * k
        elif x[1] <= 1.6:
            values[0] = 6 * k
        elif x[1] <= 2.4:
            values[0] = 1 * k
        elif x[1] <= 3.2:
            values[0] = 10 * k
        elif x[1] <= 4.0:
            values[0] = 2 * k
