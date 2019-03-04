from firedrake import *

mu0 = Constant(1.0)
k1 = Constant(1.0)
k2 = Constant(0.1)
b_factor = Constant(1.0)

# Source and gravitational terms
rhob1, rhob2 = Constant((0.0, 0.0)), Constant((0.0, 0.0))
f = Constant(0.0)


def alpha1():
    return mu0 / k1


def invalpha1():
    return 1. / alpha1()


def alpha2():
    return mu0 / k2


def invalpha2():
    return 1. / alpha2()
