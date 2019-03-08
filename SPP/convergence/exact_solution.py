from firedrake import *


def exact_solution(x, y, k, mu):
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    v_exact = -(k / mu) * grad(p_exact)
    return p_exact, v_exact
