import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc


def plot_matrix(A, **kwargs):
    """Provides a plot of a matrix."""
    fig, ax = plt.subplots(1, 1)

    petsc_mat = A.M.handle

    size = petsc_mat.getSize()
    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    my_cmap = plt.cm.magma
    my_cmap.set_bad(color="lightgray")

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return plot


def plot_matrix_mixed(A, **kwargs):
    """Provides a plot of a mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    petsc_mat = A.M.handle

    total_size = petsc_mat.getSize()
    f0_size = A.M[0, 0].handle.getSize()
    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    my_cmap = plt.cm.magma
    my_cmap.set_bad(color="lightgray")

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")

    return plot


def plot_matrix_hybrid(A, **kwargs):
    """Provides a plot of a hybrid-mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    petsc_mat = A.M.handle

    total_size = petsc_mat.getSize()
    f0_size = A.M[0, 0].handle.getSize()
    f1_size = A.M[1, 1].handle.getSize()

    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    my_cmap = plt.cm.magma
    my_cmap.set_bad(color="lightgray")

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)
    # plot = plt.spy(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")
    ax.axhline(y=f0_size[0] + f1_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] + f1_size[0] - 0.5, color="k")

    return plot


N = 5
mesh = UnitSquareMesh(N, N, quadrilateral=False)

RT = FunctionSpace(mesh, "RT", 2)
RTd = FunctionSpace(mesh, "DRT", 2)
DG = FunctionSpace(mesh, "DG", 1)
DGT = FunctionSpace(mesh, "DGT", 1)

# Individual velocity mass matrices
rt_mass = dot(TestFunction(RT), TrialFunction(RT)) * dx
rtd_mass = dot(TestFunction(RTd), TrialFunction(RTd)) * dx

rt_M = assemble(rt_mass, mat_type="aij")
rtd_M = assemble(rtd_mass, mat_type="aij")

# fig, ax = plt.subplots(1, 1)

plot_matrix(rt_M)
# plt.savefig('../figures/rt2_mass.png')
plt.show()
# plt.cla()

plot_matrix(rtd_M)
# plt.savefig('../figures/drt2_mass.png')
plt.show()
# plt.cla()

# Global sparsity patterns for mixed and hybrid-mixed methods
W = RT * DG
Wh = RTd * DG * DGT

u, p = TrialFunctions(W)
w, q = TestFunctions(W)
a = (dot(u, w) - div(w) * p + div(u) * q + p * q) * dx
A_mixed = assemble(a, mat_type="aij")

plot_matrix_mixed(A_mixed)
# plt.savefig('../figures/global_mixed_sparsity.png')
plt.show()
# plt.cla()

uh, ph, lambdar = TrialFunctions(Wh)
wh, qh, gammar = TestFunctions(Wh)
n = FacetNormal(mesh)
ah = (dot(uh, wh) - div(wh) * ph + div(uh) * qh + ph * qh) * dx + (
    jump(wh, n=n) * lambdar("+") + jump(uh, n=n) * gammar("+")
) * dS
A_hybrid = assemble(ah, mat_type="aij")

plot_matrix_hybrid(A_hybrid)
# plt.savefig('../figures/global_hybridized_sparsity.png')
plt.show()
# plt.cla()
