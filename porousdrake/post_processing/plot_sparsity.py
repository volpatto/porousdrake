"""
Must of this module was created by Thomas Gibson from Imperial College/UK.
Some parts are modified and adapted by Diego Volpatto.
"""
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc

my_cmap = plt.cm.winter
my_cmap.set_bad(color="lightgray")


def plot_matrix(a_form, bcs=[], **kwargs):
    """Provides a plot of a matrix."""
    fig, ax = plt.subplots(1, 1)

    A = assemble(a_form, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle

    size = petsc_mat.getSize()
    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return plot


def plot_matrix_mixed(a_form, bcs=[], **kwargs):
    """Provides a plot of a mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    A = assemble(a_form, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle

    total_size = petsc_mat.getSize()
    f0_size = A.M[0, 0].handle.getSize()
    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")

    return plot


def plot_matrix_hybrid_full(a_form, bcs=[], **kwargs):
    """Provides a plot of a full hybrid-mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    A = assemble(a_form, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle

    total_size = petsc_mat.getSize()
    f0_size = A.M[0, 0].handle.getSize()
    f1_size = A.M[1, 1].handle.getSize()

    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")
    ax.axhline(y=f0_size[0] + f1_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] + f1_size[0] - 0.5, color="k")

    return plot


def plot_matrix_hybrid_multiplier_spp(a_form, bcs=[], **kwargs):
    """Provides a plot of a condensed hybrid-mixed matrix for single scale problems."""
    fig, ax = plt.subplots(1, 1)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bcs)

    petsc_mat = Smat.M.handle
    total_size = petsc_mat.getSize()
    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, cmap=my_cmap, **kwargs)
    # Below there is the spy alternative
    # plot = plt.spy(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return plot
