"""cca_py — Python re-implementation of Seurat's CCA for single-cell integration.

Mirrors :func:`Seurat::RunCCA` semantics:

>>> import numpy as np
>>> from cca_py import run_cca
>>> # X: (n_features, n_cells_1), Y: (n_features, n_cells_2) — matched genes
>>> result = run_cca(X, Y, num_cc=20)
>>> result.ccv.shape
(n_cells_1 + n_cells_2, 20)

The diagonalisation is a truncated SVD of the cell × cell cross-covariance
``X^T Y`` (after row-standardising both matrices), which is the same SVD
``Seurat::RunCCA.default`` runs through ``irlba``. We use
``scipy.sparse.linalg.svds`` (LOBPCG / ARPACK) so the implementation
behaves identically on dense and sparse inputs.
"""
from __future__ import annotations

from .cca import RunCCAResult, l2_normalize, run_cca, standardize
from .anndata_adapter import run_cca_anndata

__version__ = "0.1.0"

__all__ = [
    "run_cca",
    "RunCCAResult",
    "standardize",
    "l2_normalize",
    "run_cca_anndata",
]
