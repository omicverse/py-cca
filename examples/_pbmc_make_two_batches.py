"""Prep step — load pbmc3k, split into two batches with a known batch
effect, write feature×cell CSVs that both R and Python will consume."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd

OUT = Path("/scratch/users/steorra/analysis/omicverse_dev/py-CCA/examples/_pbmc_pair")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    sc.settings.verbosity = 1
    adata = sc.read_h5ad(
        "/scratch/users/steorra/analysis/omicverse_dev/omicverse-test/notebooks/data/pbmc3k_benchmark/pbmc3k_from_10x.h5ad"
    )
    print(f"raw pbmc3k: {adata.shape}")

    # Standard Seurat-style preprocessing (matches the integration tutorial)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 15].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    print(f"after QC + log1p + HVG: {adata.shape}, HVG={adata.var['highly_variable'].sum()}")

    # Split into two batches.
    rng = np.random.default_rng(42)
    n = adata.n_obs
    perm = rng.permutation(n)
    half = n // 2
    idx_a = sorted(perm[:half])
    idx_b = sorted(perm[half:])

    adata_a = adata[idx_a].copy()
    adata_b = adata[idx_b].copy()
    adata_a.obs["batch"] = "A"
    adata_b.obs["batch"] = "B"

    # Inject a known per-gene batch effect on B (simulates a technical
    # bias across protocols). After this, PCA on the merged matrix should
    # separate the two batches by batch, not by biology — the canonical
    # CCA-fixable scenario.
    batch_effect = rng.normal(0.0, 0.6, size=adata.n_vars).astype(np.float32)
    Xb = adata_b.X.toarray() if hasattr(adata_b.X, "toarray") else adata_b.X
    Xb = Xb + batch_effect[None, :]
    Xb = np.clip(Xb, 0, None)
    adata_b.X = Xb.astype(np.float32)

    print(f"batch A: {adata_a.shape}   batch B: {adata_b.shape}")
    print(f"batch effect injected on B (per-gene shift, σ=0.6)")

    # Restrict to highly-variable genes (the standard Seurat CCA practice)
    hvg_mask = adata_a.var["highly_variable"] & adata_b.var["highly_variable"]
    print(f"shared HVG (used for CCA): {hvg_mask.sum()}")
    adata_a = adata_a[:, hvg_mask].copy()
    adata_b = adata_b[:, hvg_mask].copy()

    # Save AnnData for downstream Python-side work
    adata_a.write(OUT / "adata_a.h5ad", compression="gzip")
    adata_b.write(OUT / "adata_b.h5ad", compression="gzip")

    # Save (genes × cells) matrices for R-side Seurat::RunCCA — same layout RunCCA expects
    Xa = adata_a.X.toarray() if hasattr(adata_a.X, "toarray") else adata_a.X
    Xb = adata_b.X.toarray() if hasattr(adata_b.X, "toarray") else adata_b.X
    pd.DataFrame(Xa.T,
                 index=adata_a.var_names,
                 columns=adata_a.obs_names).to_csv(OUT / "X_A.csv")
    pd.DataFrame(Xb.T,
                 index=adata_b.var_names,
                 columns=adata_b.obs_names).to_csv(OUT / "X_B.csv")

    print(f"\nwrote to {OUT}/")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
