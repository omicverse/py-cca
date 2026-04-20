"""Smoke tests — `run_cca` returns the expected shapes and orientations."""
from __future__ import annotations

import numpy as np
import pytest

from cca_py import run_cca, standardize, l2_normalize


def _make_pair(n_features=200, n1=80, n2=120, n_shared=10, seed=0):
    rng = np.random.default_rng(seed)
    # Latent shared structure across n_shared dimensions, plus dataset-specific noise
    Z1 = rng.standard_normal((n_shared, n1))
    Z2 = rng.standard_normal((n_shared, n2))
    W = rng.standard_normal((n_features, n_shared))
    X = W @ Z1 + 0.1 * rng.standard_normal((n_features, n1))
    Y = W @ Z2 + 0.1 * rng.standard_normal((n_features, n2))
    return X, Y


def test_shapes():
    X, Y = _make_pair()
    res = run_cca(X, Y, num_cc=10)
    assert res.ccv.shape == (X.shape[1] + Y.shape[1], 10)
    assert res.d.shape == (10,)
    assert res.n1 == X.shape[1]
    assert res.n2 == Y.shape[1]


def test_singular_values_descending():
    X, Y = _make_pair()
    res = run_cca(X, Y, num_cc=15)
    # SVs should be sorted descending
    assert np.all(np.diff(res.d) <= 1e-9)
    # All SVs should be positive
    assert np.all(res.d > 0)


def test_split_returns_two_halves():
    X, Y = _make_pair(n1=50, n2=70)
    res = run_cca(X, Y, num_cc=8)
    u, v = res.split()
    assert u.shape == (50, 8)
    assert v.shape == (70, 8)
    np.testing.assert_array_equal(np.vstack([u, v]), res.ccv)


def test_sign_flip_makes_first_row_nonneg():
    X, Y = _make_pair()
    res = run_cca(X, Y, num_cc=10, sign_flip=True)
    # First row of ccv should have all non-negative entries
    assert np.all(res.ccv[0] >= 0)


def test_seed_determinism():
    X, Y = _make_pair()
    a = run_cca(X, Y, num_cc=10, seed=7)
    b = run_cca(X, Y, num_cc=10, seed=7)
    np.testing.assert_allclose(a.ccv, b.ccv, atol=1e-10)
    np.testing.assert_allclose(a.d,   b.d,   atol=1e-10)


def test_recovers_shared_subspace():
    """If two datasets share a 5-dim latent, the top 5 CCs should
    capture it cleanly — measured by correlation of ccv[:n1, :5]
    with the true shared latent Z1."""
    rng = np.random.default_rng(42)
    n_shared = 5
    n_features, n1, n2 = 300, 100, 150
    Z1 = rng.standard_normal((n_shared, n1))
    Z2 = rng.standard_normal((n_shared, n2))
    W = rng.standard_normal((n_features, n_shared))
    X = W @ Z1 + 0.05 * rng.standard_normal((n_features, n1))
    Y = W @ Z2 + 0.05 * rng.standard_normal((n_features, n2))

    res = run_cca(X, Y, num_cc=n_shared)
    u = res.ccv[: res.n1]  # (n1, 5)

    # Pick the best linear combo of u that recovers Z1[0]
    # — equivalent to checking that the column spans match.
    M = u @ np.linalg.pinv(u) @ Z1.T  # project Z1.T into span(u) and back
    # cosine between projected and original should be ~1
    for j in range(n_shared):
        true_z = Z1[j]
        cos = abs(true_z @ M[:, j] / (np.linalg.norm(true_z) * np.linalg.norm(M[:, j]) + 1e-12))
        assert cos > 0.9, f"shared dim {j} not recovered (cos={cos:.3f})"


def test_standardize_zeroes_constant_columns():
    """A column with zero variance must standardise to all zeros (no NaN).

    Seurat's Standardize z-scores per column (per cell), so a cell with
    flat expression across genes should fall out as zeros.
    """
    # Column 0 is constant across rows; column 1 varies.
    M = np.array([[7.0, 1.0],
                  [7.0, 2.0],
                  [7.0, 3.0],
                  [7.0, 4.0],
                  [7.0, 5.0]])
    out = standardize(M)
    assert np.allclose(out[:, 0], 0.0)          # constant col → zeros, no NaN
    assert np.allclose(out[:, 1].mean(), 0.0)   # varying col → zero mean
    assert np.allclose(out[:, 1].std(ddof=1), 1.0)


def test_l2_normalize():
    M = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = l2_normalize(M, axis=1)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)


def test_feature_mismatch_raises():
    X = np.zeros((100, 50))
    Y = np.zeros((90, 50))
    with pytest.raises(ValueError, match="same number of features"):
        run_cca(X, Y, num_cc=5)


def test_sparse_input_handled():
    """Sparse inputs should be auto-densified and produce the same result."""
    import scipy.sparse as sp

    X, Y = _make_pair(n_features=100, n1=40, n2=60)
    Xs = sp.csr_matrix(X)
    Ys = sp.csr_matrix(Y)

    a = run_cca(X, Y, num_cc=10)
    b = run_cca(Xs, Ys, num_cc=10)
    # Up to sign (sign_flip should normalise it) the embeddings should match
    np.testing.assert_allclose(a.d, b.d, rtol=1e-6)
