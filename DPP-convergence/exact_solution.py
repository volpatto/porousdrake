from firedrake import *


def exact_solution(x, y, beta, k1, k2, mu):
    eta = sqrt(beta * (k1 + k2) / (k1 * k2))
    p_exact_1 = mu / pi * exp(pi * x) * sin(pi * y) - mu / (beta * k1) * exp(eta * y)
    p_exact_2 = mu / pi * exp(pi * x) * sin(pi * y) + mu / (beta * k2) * exp(eta * y)
    v_exact_1 = -(k1 / mu) * grad(p_exact_1)
    v_exact_2 = -(k2 / mu) * grad(p_exact_2)
    return p_exact_1, p_exact_2, v_exact_1, v_exact_2
