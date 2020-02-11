from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
from porousdrake.SPP.velocity_patch.model_parameters import *

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")


def sdhm(
    mesh,
    degree,
    beta_0=Constant(1e-2),
    delta_0=Constant(1.0),
    delta_1=Constant(-0.5),
    delta_2=Constant(0.5),
    delta_3=Constant(0.5),
    mesh_parameter=True,
    solver_parameters={},
):
    if not solver_parameters:
        solver_parameters = {
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "pmat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            # Use the static condensation PC for hybridized problems
            # and use a direct solve on the reduced system for lambda_h
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
            "condensed_field": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        }

    pressure_family = "DG"
    velocity_family = "DG"
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    DPP_solution = Function(W)
    u, p, lambda_h = split(DPP_solution)
    v, q, mu_h = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Permeability
    k = conditional(
        y <= 0.8,
        80 * k_ref,
        conditional(
            y <= 1.6,
            30 * k_ref,
            conditional(y <= 2.4, 5 * k_ref, conditional(y <= 3.2, 50 * k_ref, 10 * k_ref)),
        ),
    )

    def alpha():
        return mu / k

    def invalpha():
        return 1.0 / alpha()

    # Flux BCs
    un_1 = -k / mu
    un_2 = k / mu

    # Stabilizing parameter
    beta = beta_0 / h
    beta_avg = beta_0 / h("+")
    if mesh_parameter:
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h

    # Mixed classical terms
    a = (dot(alpha() * u, v) - div(v) * p - delta_0 * q * div(u)) * dx
    L = -delta_0 * f * q * dx - delta_0 * dot(rhob, v) * dx
    # Stabilizing terms
    ###
    a += delta_1 * inner(invalpha() * (alpha() * u + grad(p)), delta_0 * alpha() * v + grad(q)) * dx
    ###
    a += delta_2 * alpha() * div(u) * div(v) * dx
    L += delta_2 * alpha() * f * div(v) * dx
    ###
    a += delta_3 * inner(invalpha() * curl(alpha() * u), curl(alpha() * v)) * dx
    # Hybridization terms
    ###
    a += lambda_h("+") * jump(v, n) * dS + mu_h("+") * jump(u, n) * dS
    ###
    a += beta_avg * invalpha()("+") * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
    # Weakly imposed BC from hybridization
    a += (lambda_h * dot(v, n) + mu_h * (dot(u, n) - un_1)) * ds(1)
    a += (lambda_h * dot(v, n) + mu_h * (dot(u, n) - un_2)) * ds(2)
    a += (lambda_h * dot(v, n) + mu_h * (dot(u, n))) * (ds(3) + ds(4))

    F = a - L

    #  Solving SC below
    PETSc.Sys.Print(
        "*******************************************\nSolving using static condensation.\n"
    )
    problem_flow = NonlinearVariationalProblem(F, DPP_solution)
    solver_flow = NonlinearVariationalSolver(problem_flow, solver_parameters=solver_parameters)
    solver_flow.solve()

    # Returning numerical and exact solutions
    p_sol, v_sol = _decompose_numerical_solution_hybrid(DPP_solution)
    return p_sol, v_sol


def lsh(
    mesh,
    degree,
    beta_0=Constant(1e-2),
    delta_0=Constant(1.0),
    delta_1=Constant(-0.5),
    delta_2=Constant(0.5),
    delta_3=Constant(0.5),
    mesh_parameter=True,
    solver_parameters={},
):
    if not solver_parameters:
        solver_parameters = {
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "pmat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            # Use the static condensation PC for hybridized problems
            # and use a direct solve on the reduced system for lambda_h
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
            "condensed_field": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        }

    pressure_family = "DG"
    velocity_family = "DG"
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    DPP_solution = Function(W)
    u, p, lambda_h = split(DPP_solution)
    v, q, mu_h = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Permeability
    k = conditional(
        y <= 0.8,
        80 * k_ref,
        conditional(
            y <= 1.6,
            30 * k_ref,
            conditional(y <= 2.4, 5 * k_ref, conditional(y <= 3.2, 50 * k_ref, 10 * k_ref)),
        ),
    )

    def alpha():
        return mu / k

    def invalpha():
        return 1.0 / alpha()

    # Flux BCs
    un_1 = -k / mu
    un_2 = k / mu

    # Stabilizing parameter
    beta = beta_0 / h
    beta_avg = beta_0 / h("+")
    if mesh_parameter:
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h

    # Mixed classical terms (TODO: include gravity term)
    a = (
        inner(alpha() * u, v + invalpha() * grad(q))
        - div(v) * p
        + inner(grad(p), invalpha() * grad(q))
    ) * dx
    # Stabilizing terms
    ###
    # a += inner(invalpha() * (alpha() * u + grad(p)), grad(q)) * dx
    ###
    a += div(u) * div(v) * dx
    L = f * div(v) * dx
    ###
    # a += inner(curl(alpha() * u), curl(alpha() * v)) * dx
    # Hybridization terms
    ###
    a += lambda_h("+") * jump(v, n) * dS + mu_h("+") * jump(u, n) * dS
    ###
    a += beta_avg * (lambda_h("+") - p("+")) * (mu_h("+")) * dS
    # Weakly imposed BC from hybridization
    a += (lambda_h * dot(v, n) + mu_h * (dot(u, n) - un_1)) * ds(1)
    a += (lambda_h * dot(v, n) + mu_h * (dot(u, n) - un_2)) * ds(2)
    a += (lambda_h * dot(v, n) + mu_h * (dot(u, n))) * (ds(3) + ds(4))

    F = a - L

    #  Solving SC below
    PETSc.Sys.Print(
        "*******************************************\nSolving using static condensation.\n"
    )
    problem_flow = NonlinearVariationalProblem(F, DPP_solution)
    solver_flow = NonlinearVariationalSolver(problem_flow, solver_parameters=solver_parameters)
    solver_flow.solve()

    # Returning numerical and exact solutions
    p_sol, v_sol = _decompose_numerical_solution_hybrid(DPP_solution)
    return p_sol, v_sol


def dgls(
    mesh,
    degree,
    delta_0=Constant(1.0),
    delta_1=Constant(-0.5),
    delta_2=Constant(0.5),
    delta_3=Constant(0.5),
    eta_p=Constant(0.0),
    eta_u=Constant(1.0),
    mesh_parameter=True,
    solver_parameters={},
):
    if not solver_parameters:
        solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    pressure_family = "DG"
    velocity_family = "DG"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    DPP_solution = Function(W)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Permeability
    k = conditional(
        y <= 0.8,
        80 * k_ref,
        conditional(
            y <= 1.6,
            30 * k_ref,
            conditional(y <= 2.4, 5 * k_ref, conditional(y <= 3.2, 50 * k_ref, 10 * k_ref)),
        ),
    )

    def alpha():
        return mu / k

    def invalpha():
        return 1.0 / alpha()

    # Flux BCs
    un_1 = -k / mu
    un_2 = k / mu

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0
    if mesh_parameter:
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h

    # Mixed classical terms
    a = (dot(alpha() * u, v) - div(v) * p - delta_0 * q * div(u)) * dx
    L = -delta_0 * f * q * dx - delta_0 * dot(rhob, v) * dx
    # DG terms
    a += jump(v, n) * avg(p) * dS - avg(q) * jump(u, n) * dS
    # Edge stabilizing terms
    a += (eta_u * h_avg) * avg(alpha()) * (jump(u, n) * jump(v, n)) * dS + (eta_p / h_avg) * avg(
        1.0 / alpha()
    ) * dot(jump(q, n), jump(p, n)) * dS
    # Volume stabilizing terms
    ###
    a += delta_1 * inner(invalpha() * (alpha() * u + grad(p)), delta_0 * alpha() * v + grad(q)) * dx
    ###
    a += delta_2 * alpha() * div(u) * div(v) * dx
    L += delta_2 * alpha() * f * div(v) * dx
    ###
    a += delta_3 * inner(invalpha() * curl(alpha() * u), curl(alpha() * v)) * dx
    # Weakly imposed BC by Nitsche's method
    a += dot(v, n) * p * ds - q * dot(u, n) * ds
    L += (
        -q * un_1 * ds(1)
        - q * un_2 * ds(2)
        - delta_1 * dot(delta_0 * alpha() * v + grad(q), invalpha() * rhob) * dx
    )
    a += eta_u / h * inner(dot(v, n), dot(u, n)) * ds
    L += eta_u / h * dot(v, n) * un_1 * ds(1) + eta_u / h * dot(v, n) * un_2 * ds(2)

    #  Solving
    problem_flow = LinearVariationalProblem(a, L, DPP_solution, bcs=[], constant_jacobian=False)
    solver_flow = LinearVariationalSolver(
        problem_flow, options_prefix="dpp_flow", solver_parameters=solver_parameters
    )
    solver_flow.solve()

    # Returning numerical and exact solutions
    p_sol, v_sol = _decompose_numerical_solution_mixed(DPP_solution)
    return p_sol, v_sol


def cgls(
    mesh,
    degree,
    delta_0=Constant(1.0),
    delta_1=Constant(-0.5),
    delta_2=Constant(0.5),
    delta_3=Constant(0.5),
    eta_u=Constant(0),
    mesh_parameter=True,
    solver_parameters={},
):
    if not solver_parameters:
        solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    pressure_family = "CG"
    velocity_family = "CG"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    DPP_solution = Function(W)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Permeability
    k = conditional(
        y <= 0.8,
        80 * k_ref,
        conditional(
            y <= 1.6,
            30 * k_ref,
            conditional(y <= 2.4, 5 * k_ref, conditional(y <= 3.2, 50 * k_ref, 10 * k_ref)),
        ),
    )

    def alpha():
        return mu / k

    def invalpha():
        return 1.0 / alpha()

    #  Flux BCs
    un_1 = -k / mu
    un_2 = k / mu

    # Mesh stabilizing parameter
    if mesh_parameter:
        delta_2 = delta_2 * h * h
        delta_3 = delta_3 * h * h

    # Mixed classical terms
    a = (dot(alpha() * u, v) - div(v) * p - delta_0 * q * div(u)) * dx
    L = -delta_0 * f * q * dx - delta_0 * dot(rhob, v) * dx
    # Volume stabilizing terms
    ###
    a += delta_1 * inner(invalpha() * (alpha() * u + grad(p)), delta_0 * alpha() * v + grad(q)) * dx
    ###
    a += delta_2 * alpha() * div(u) * div(v) * dx
    L += delta_2 * alpha() * f * div(v) * dx
    ###
    a += delta_3 * inner(invalpha() * curl(alpha() * u), curl(alpha() * v)) * dx
    # Weakly imposed BC by Nitsche's method
    a += dot(v, n) * p * ds - q * dot(u, n) * ds
    L += -q * un_1 * ds(1) - q * un_2 * ds(2)
    a += eta_u / h * inner(dot(v, n), dot(u, n)) * ds
    L += eta_u / h * dot(v, n) * un_1 * ds(1) + eta_u / h * dot(v, n) * un_2 * ds(2)

    #  Solving
    problem_flow = LinearVariationalProblem(a, L, DPP_solution, bcs=[], constant_jacobian=False)
    solver_flow = LinearVariationalSolver(
        problem_flow, options_prefix="dpp_flow", solver_parameters=solver_parameters
    )
    solver_flow.solve()

    # Returning numerical and exact solutions
    p_sol, v_sol = _decompose_numerical_solution_mixed(DPP_solution)
    return p_sol, v_sol


def _decompose_numerical_solution_hybrid(solution):
    v_sol = solution.sub(0)
    v_sol.rename("Velocity", "label")
    p_sol = solution.sub(1)
    p_sol.rename("Pressure", "label")
    return p_sol, v_sol


def _decompose_numerical_solution_mixed(solution):
    v_sol = solution.sub(0)
    v_sol.rename("Velocity", "label")
    p_sol = solution.sub(1)
    p_sol.rename("Pressure", "label")
    return p_sol, v_sol
