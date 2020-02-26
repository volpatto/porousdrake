from firedrake import *

# Stabilizing parameters
delta_0 = Constant(1)
delta_1 = Constant(-0.5)
delta_2 = Constant(0.5)
delta_3 = Constant(0.5)
eta_p = Constant(1e0)  # * eta_u
eta_u = Constant(1e1) * eta_p
beta_0 = Constant(1.0e0)
stabilizing_mass_constant = Constant(0)
ls_lambda_constant = Constant(0)
mesh_parameter = True

solvers_args = {
    "cgls_full": {
        "delta_0": Constant(1),
        "delta_1": Constant(-0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
    },
    "cgls_div": {
        "delta_0": Constant(1),
        "delta_1": Constant(-0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
    },
    "mgls_full": {
        "delta_0": Constant(1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
    },
    "mgls": {
        "delta_0": Constant(1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
    },
    "mvh_full": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
    },
    "mvh_div": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
    },
    "mvh": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.0),
        "delta_3": Constant(0.0),
    },
    ###############################################
    "dgls_full": {
        "delta_0": Constant(1),
        "delta_1": Constant(-0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    "dgls_div": {
        "delta_0": Constant(1),
        "delta_1": Constant(-0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    "dmgls_full": {
        "delta_0": Constant(1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    "dmgls": {
        "delta_0": Constant(1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    "dmvh_full": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    "dmvh_div": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    "dmvh": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.0),
        "delta_3": Constant(0.0),
        "eta_u": eta_u,
        "eta_p": eta_p,
    },
    ###############################################
    "sdhm_full": {
        "delta_0": Constant(1),
        "delta_1": Constant(-0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
        "beta_0": beta_0,
    },
    "sdhm_div": {
        "delta_0": Constant(1),
        "delta_1": Constant(-0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
        "beta_0": beta_0,
    },
    "hmgls_full": {
        "delta_0": Constant(1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
        "beta_0": beta_0,
    },
    "hmgls": {
        "delta_0": Constant(1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
        "beta_0": beta_0,
    },
    "hmvh_full": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.5),
        "beta_0": beta_0,
    },
    "hmvh_div": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.5),
        "delta_3": Constant(0.0),
        "beta_0": beta_0,
    },
    "hmvh": {
        "delta_0": Constant(-1),
        "delta_1": Constant(0.5),
        "delta_2": Constant(0.0),
        "delta_3": Constant(0.0),
        "beta_0": beta_0,
    },
    "lsh": {
        "beta_0": beta_0,
        "stabilizing_mass_constant": Constant(0),
        "ls_lambda_constant": Constant(0),
    },
    "lsh_mass": {
        "beta_0": beta_0,
        "stabilizing_mass_constant": Constant(1),
        "ls_lambda_constant": Constant(0),
    },
    "lsh_lambda": {
        "beta_0": beta_0,
        "stabilizing_mass_constant": Constant(0),
        "ls_lambda_constant": Constant(1),
    },
    "lsh_full": {
        "beta_0": beta_0,
        "stabilizing_mass_constant": Constant(1),
        "ls_lambda_constant": Constant(1),
    },
    "dls": {"eta_u": eta_u, "eta_p": eta_p, "curl_weight": Constant(1)},
    "clsq": dict(),
}
