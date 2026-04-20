"""Compare py-CCA against Seurat::RunCCA on PBMC two-batch fixture and
produce side-by-side figures suitable for ImageStore/pytrans/."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from cca_py import run_cca, l2_normalize

DIR = Path("/scratch/users/steorra/analysis/omicverse_dev/py-CCA/examples/_pbmc_pair")
OUT = Path("/tmp/ImageStore/pytrans")
OUT.mkdir(parents=True, exist_ok=True)


def _to_arr(v):
    return np.array(
        [np.nan if isinstance(x, str) and x.upper() == "NA" else x for x in v],
        dtype=np.float64,
    )


# 1) Load fixtures
adata_a = ad.read_h5ad(DIR / "adata_a.h5ad")
adata_b = ad.read_h5ad(DIR / "adata_b.h5ad")
seurat = json.loads((DIR / "seurat_cca.json").read_text())
print(f"A: {adata_a.shape}   B: {adata_b.shape}")
print(f"Seurat ccv: {seurat['ccv_dim']}, d[0..4] = {seurat['d'][:5]}")

# 2) Run py-CCA on the same data — match Seurat layout: features × cells
def _matrix(adata):
    M = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    return np.asarray(M, dtype=np.float64).T  # AnnData: cells × genes → genes × cells

X = _matrix(adata_a)
Y = _matrix(adata_b)
py = run_cca(X, Y, num_cc=30, seed=42)
print(f"py ccv shape: {py.ccv.shape}, d[0..4] = {py.d[:5]}")

# 3) Compute parity metrics
r_d = _to_arr(seurat["d"])
r_ccv = _to_arr(seurat["ccv"]).reshape(tuple(seurat["ccv_dim"]), order="F")

rel_err_d = np.abs(py.d - r_d) / np.maximum(np.abs(r_d), 1e-12)
print(f"\nsingular values: max relative error = {rel_err_d.max():.3e}")
print(f"singular values: median relative error = {np.median(rel_err_d):.3e}")

# Per-CC absolute correlation (sign-invariant)
corrs = np.array([
    abs(np.corrcoef(py.ccv[:, j], r_ccv[:, j])[0, 1]) for j in range(30)
])
print(f"per-CC |corr|: min = {corrs.min():.4f}, mean = {corrs.mean():.4f}")

# Subspace match
Qp, _ = np.linalg.qr(py.ccv)
Qr, _ = np.linalg.qr(r_ccv)
subspace_diff = np.linalg.norm(Qp @ Qp.T - Qr @ Qr.T) / np.sqrt(30)
print(f"subspace ||P_py − P_r||_F / √k = {subspace_diff:.3e}")


# 4) Build a merged AnnData with batch labels; compute UMAPs in three spaces:
#    - raw HVG-PCA (no integration; should show batch separation)
#    - Seurat CCA
#    - py-CCA
adata_a.obs["batch"] = "A"
adata_b.obs["batch"] = "B"
merged = ad.concat([adata_a, adata_b], axis=0, join="inner")
merged.obs["batch"] = pd.Categorical(merged.obs["batch"], categories=["A", "B"])

# raw PCA
sc.pp.scale(merged, max_value=10)
sc.tl.pca(merged, n_comps=30, random_state=0)

# pyCCA embedding (already cells × CC)
u_py, v_py = py.split()
merged.obsm["X_pyCCA"] = np.vstack([u_py, v_py])

# Seurat CCA embedding
merged.obsm["X_seurat_CCA"] = np.ascontiguousarray(r_ccv)

# L2 normalize for downstream UMAP (Seurat does L2CCA before nearest-neighbor work)
merged.obsm["X_pyCCA_l2"] = l2_normalize(merged.obsm["X_pyCCA"], axis=1)
merged.obsm["X_seurat_CCA_l2"] = l2_normalize(merged.obsm["X_seurat_CCA"], axis=1)

# Compute UMAP per representation
print("\ncomputing UMAPs ...")
for rep, label in [("X_pca", "raw PCA"),
                   ("X_pyCCA_l2", "py-CCA (L2)"),
                   ("X_seurat_CCA_l2", "Seurat CCA (L2)")]:
    sc.pp.neighbors(merged, n_neighbors=15, use_rep=rep, key_added=label.replace(" ", "_"))
    sc.tl.umap(merged, neighbors_key=label.replace(" ", "_"), random_state=0)
    merged.obsm[f"X_umap_{label}"] = merged.obsm["X_umap"].copy()
    print(f"  {label}: done")


# ─────────────────────────────────────── Figure 1: integration UMAP triptych
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
palette = {"A": "#4878D0", "B": "#EE854A"}
for ax, label in zip(axes, ["raw PCA", "Seurat CCA (L2)", "py-CCA (L2)"]):
    coords = merged.obsm[f"X_umap_{label}"]
    for batch in ["A", "B"]:
        m = merged.obs["batch"] == batch
        ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.7,
                   c=palette[batch], label=f"batch {batch} (n={m.sum()})",
                   edgecolor="none")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=9, markerscale=3)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(
    "PBMC3k split into two batches with synthetic per-gene shift (σ=0.6) — UMAP per representation\n"
    "Raw PCA: batches separate (batch effect dominates).  "
    "Seurat CCA / py-CCA: batches mix (CCA aligned the shared subspace)",
    fontsize=11,
)
plt.tight_layout()
fp = OUT / "py-CCA_pbmc_integration_umap.png"
plt.savefig(fp, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nwrote {fp.name}  ({fp.stat().st_size/1024:.0f} KB)")


# ─────────────────────────────────────── Figure 2: parity (singular values + per-CC correlation + scatter)
fig = plt.figure(figsize=(15, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])

ax = fig.add_subplot(gs[0, 0])
k = np.arange(1, 31)
ax.plot(k, r_d, "o-", color="#4878D0", lw=1.5, ms=7, alpha=0.75, label="Seurat (R)")
ax.plot(k, py.d, "x", color="#EE854A", ms=10, mew=2.0, alpha=0.95, label="py-CCA")
ax.set_xlabel("Component (CC)")
ax.set_ylabel("Singular value")
ax.set_title("Singular values — overlay")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.bar(k, corrs, color="#4878D0", alpha=0.85)
ax.axhline(0.999, color="red", ls="--", lw=1, label="0.999 floor")
ax.set_xlabel("Component (CC)")
ax.set_ylabel("|corr(py CC, Seurat CC)|")
ax.set_title(f"Per-component embedding correlation\n(min = {corrs.min():.4f}, mean = {corrs.mean():.4f})")
ax.set_ylim(0.99, 1.001)
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(r_d, py.d, color="#4878D0", s=60, alpha=0.85, edgecolor="white", lw=0.5)
lo, hi = r_d.min() * 0.98, r_d.max() * 1.02
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
ax.set_xlabel("Seurat singular value")
ax.set_ylabel("py-CCA singular value")
ax.set_title(f"Singular values — scatter\n(max rel err = {rel_err_d.max():.2e})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", "box")

fig.suptitle(
    "py-CCA vs Seurat::RunCCA on PBMC3k two-batch fixture\n"
    f"(num_cc=30, n_features=2000 HVG, n_cells = {py.n1 + py.n2})",
    fontsize=11,
)
plt.tight_layout()
fp = OUT / "py-CCA_pbmc_parity.png"
plt.savefig(fp, dpi=120, bbox_inches="tight")
plt.close()
print(f"wrote {fp.name}  ({fp.stat().st_size/1024:.0f} KB)")


# ─────────────────────────────────────── Figure 3: cell-by-cell embedding heatmap, py vs R
# Row-correlate py.ccv and r_ccv per cell to show full embedding agreement.
cell_corr = np.array([
    abs(np.corrcoef(py.ccv[i], r_ccv[i])[0, 1]) for i in range(py.ccv.shape[0])
])
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(cell_corr, bins=40, color="#4878D0", alpha=0.85, edgecolor="white")
ax.axvline(0.999, color="red", ls="--", lw=1, label="0.999")
ax.set_xlabel("|corr(py-CCA cell embedding, Seurat cell embedding)|")
ax.set_ylabel("# cells")
ax.set_title(
    "Per-cell agreement between py-CCA and Seurat::RunCCA embeddings\n"
    f"({py.n1 + py.n2} cells; min={cell_corr.min():.4f}, "
    f"median={np.median(cell_corr):.4f}, max={cell_corr.max():.4f})"
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fp = OUT / "py-CCA_pbmc_cell_corr.png"
plt.savefig(fp, dpi=120, bbox_inches="tight")
plt.close()
print(f"wrote {fp.name}  ({fp.stat().st_size/1024:.0f} KB)")


# Summary numbers
print("\n=== Headline numbers ===")
print(f"  num cells (n1 + n2)       : {py.n1 + py.n2}")
print(f"  num shared HVG features   : {X.shape[0]}")
print(f"  num CC components         : 30")
print(f"  max rel SV error          : {rel_err_d.max():.2e}")
print(f"  median rel SV error       : {np.median(rel_err_d):.2e}")
print(f"  min per-CC |corr|         : {corrs.min():.4f}")
print(f"  mean per-CC |corr|        : {corrs.mean():.4f}")
print(f"  min per-cell |corr|       : {cell_corr.min():.4f}")
print(f"  median per-cell |corr|    : {np.median(cell_corr):.4f}")
print(f"  subspace diff (k=30)      : {subspace_diff:.3e}")
