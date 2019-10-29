from firedrake import *

mu = Constant(1.0)
k = Constant(1.0)

# Source and gravitational terms
rhob = Constant((0.0, 0.0))
f = Constant(0.0)


def alpha():
    return mu / k


def invalpha():
    return 1.0 / alpha()
