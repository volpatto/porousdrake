from firedrake import *
import os
from DPP.convergence.solvers import decompose_exact_solution


def write_pvd_mixed_formulations(name, mesh, degree, p1_sol, v1_sol, p2_sol, v2_sol):
    p_e_1, v_e_1, p_e_2, v_e_2 = decompose_exact_solution(mesh, degree)
    p_e_1.rename('Exact macro pressure', 'label')
    p_e_2.rename('Exact micro pressure', 'label')
    v_e_1.rename('Exact macro velocity', 'label')
    v_e_2.rename('Exact micro velocity', 'label')
    v1_sol.rename('Macro velocity', 'label')
    p1_sol.rename('Macro pressure', 'label')
    v2_sol.rename('Micro velocity', 'label')
    p2_sol.rename('Micro pressure', 'label')
    os.makedirs("results_%s" % name, exist_ok=True)
    output_file = File('pvd_results_%s/%s.pvd' % (name, name))
    output_file.write(p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2)
    return


def write_pvd_hybrid_formulations(name, mesh, degree, p1_sol, v1_sol, p2_sol, v2_sol):
    p_e_1, v_e_1, p_e_2, v_e_2 = decompose_exact_solution(mesh, degree)
    p_e_1.rename('Exact macro pressure', 'label')
    p_e_2.rename('Exact micro pressure', 'label')
    v_e_1.rename('Exact macro velocity', 'label')
    v_e_2.rename('Exact micro velocity', 'label')
    v1_sol.rename('Macro velocity', 'label')
    p1_sol.rename('Macro pressure', 'label')
    v2_sol.rename('Micro velocity', 'label')
    p2_sol.rename('Micro pressure', 'label')
    os.makedirs("results_%s" % name, exist_ok=True)
    output_file = File('pvd_results_%s/%s.pvd' % (name, name))
    output_file.write(p1_sol, v1_sol, p2_sol, v2_sol, p_e_1, v_e_1, p_e_2, v_e_2)
    return
