"""Time py-CCA on the same fixtures Seurat used."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cca_py import run_cca

DIR = Path("/scratch/users/steorra/analysis/omicverse_dev/py-CCA/examples/_pbmc_pair")


def _load_pbmc():
    A = pd.read_csv(DIR / "X_A.csv", index_col=0).values
    B = pd.read_csv(DIR / "X_B.csv", index_col=0).values
    return A, B


def _make_pair(n_features, n1, n2, n_shared=20, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_shared, n1))
    Z2 = rng.standard_normal((n_shared, n2))
    W = rng.standard_normal((n_features, n_shared))
    X = W @ Z1 + noise * rng.standard_normal((n_features, n1))
    Y = W @ Z2 + noise * rng.standard_normal((n_features, n2))
    return X, Y


def time_one(label, X, Y, num_cc, method, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = run_cca(X, Y, num_cc=num_cc, seed=42, method=method)
        times.append(time.perf_counter() - t0)
    best = min(times)
    median = float(np.median(times))
    print(f"[{label}] num_cc={num_cc}  shape={X.shape}+{Y.shape}  "
          f"method={method}  best {best:.3f}s (median {median:.3f}s over {n_repeats} runs)")
    return {"label": label, "num_cc": num_cc, "method": method,
            "n1": X.shape[1], "n2": Y.shape[1], "n_features": X.shape[0],
            "times_seconds": times, "best_seconds": best, "median_seconds": median}


def main():
    np.random.seed(42)
    results = []

    # 1) PBMC
    A, B = _load_pbmc()
    print(f"=== PBMC ({A.shape[0]} features × {A.shape[1]} + {B.shape[1]} cells) ===")
    for method in ["arpack", "fast", "randomized"]:
        results.append(time_one("pbmc", A, B, 30, method))

    # 2) Synthetic, scaled
    for cfg in [
        ("small_2k_500", 2000, 250, 250),
        ("med_2k_2k",    2000, 1000, 1000),
        ("large_2k_10k", 2000, 5000, 5000),
    ]:
        label, n_features, n1, n2 = cfg
        X, Y = _make_pair(n_features, n1, n2)
        print(f"\n=== {label} ===")
        for method in ["arpack", "fast", "randomized"]:
            results.append(time_one(label, X, Y, 30, method))

    out = DIR / "bench_pycca.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
