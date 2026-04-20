# py-CCA

A **pure-Python re-implementation of Seurat's `RunCCA`** (Stuart, Butler, Hoffman, Hafemeister et al., *Cell* 2019) — canonical correlation analysis for single-cell integration. Drop-in for the scanpy / AnnData ecosystem.

- AnnData-native — feeds directly into Scanpy / OmicVerse pipelines
- No `rpy2`, no R install, no Rcpp toolchain
- Numerical parity with `Seurat::RunCCA` validated across 9 (size × num_cc) configurations: singular values match to ~1e-7, subspaces match to ~1e-3 (rotation within near-degenerate eigenspaces is the only source of difference)

> Same upstream-mirror pattern as [`pymclustR`](https://pypi.org/project/pymclustR/), [`monocle2-py`](https://github.com/omicverse/py-monocle2), [`milor-py`](https://github.com/omicverse/py-miloR): the canonical implementation lives in [`omicverse`](https://github.com/Starlitnightly/omicverse); this repo is the standalone slice for users who want CCA without the full omicverse stack.

## Install

```bash
pip install py-CCA
```

## Quick-start

```python
import numpy as np
from cca_py import run_cca

# X, Y are (n_features, n_cells) matrices with matched genes
X = np.random.randn(2000, 500)   # batch 1: 2000 genes × 500 cells
Y = np.random.randn(2000, 700)   # batch 2: 2000 genes × 700 cells

result = run_cca(X, Y, num_cc=30)
print(result.ccv.shape)          # (1200, 30) — shared CC embedding
print(result.d.shape)            # (30,)      — singular values

u, v = result.split()             # per-batch halves: (500, 30) and (700, 30)
```

### AnnData adapter

```python
from cca_py import run_cca_anndata

# adata1, adata2 are scanpy AnnData objects (cells × genes)
result = run_cca_anndata(adata1, adata2, num_cc=30, layer="log1p")

# adata1.obsm['X_cca'] now holds the (n_obs_1, 30) shared embedding
# adata2.obsm['X_cca'] holds the (n_obs_2, 30) embedding for the second batch
# adata.uns['cca'] carries the singular-value diagnostics
```

## Algorithm

Direct port of `Seurat::RunCCA.default` (Seurat `R/dimensional_reduction.R`, lines 506–541):

```r
object1 <- Standardize(object1)        # z-score per cell (column)
object2 <- Standardize(object2)
mat3    <- crossprod(object1, object2) # cells_1 × cells_2 cross-cov
cca.svd <- irlba(mat3, nv = num.cc)    # truncated SVD
ccv     <- rbind(cca.svd$u, cca.svd$v) # (n1 + n2) × num.cc
# sign-flip each column so its first entry is non-negative
return(list(ccv = ccv, d = cca.svd$d))
```

We use `scipy.sparse.linalg.svds` (ARPACK) in place of `irlba`. Both are Lanczos-based and produce numerically equivalent top-k SVD truncations.

> ⚠️ **Standardize gotcha**: Seurat's `Standardize` (in `src/data_manipulation.cpp`) z-scores per **column** (per cell), not per row (per gene) — a non-obvious choice that's load-bearing for CCA correctness. We replicated it.

## Module map

| Module | What it covers |
|---|---|
| `cca_py.cca` | core `run_cca()` + `standardize()` + `l2_normalize()` |
| `cca_py.anndata_adapter` | `run_cca_anndata()` for the scanpy / AnnData ecosystem |

## Seurat parity

`tests/r_parity_dump.R` runs `Seurat::RunCCA` on three synthetic dataset sizes (small / medium / large) at three `num_cc` values (5 / 10 / 20). `tests/test_r_parity.py` then runs `py-CCA`'s `run_cca` on the same inputs and asserts:

| Quantity | Tolerance |
|---|---:|
| singular values (per-component relative error) | < **1e-5** |
| per-component embedding correlation | > **0.999** |
| Frobenius distance between the two column-span projectors | < **5e-3** |

All 9 configurations × 2 assertion families = 18 parity tests pass. To reproduce:

```bash
# in CMAP env (R + Seurat)
Rscript tests/r_parity_dump.R

# then in omicdev env
pytest tests/ -v
```

## Roadmap

This first release covers the **core SVD step** of `RunCCA`. The full Seurat integration workflow uses CCA as the first step in `FindIntegrationAnchors`:

1. ✅ `RunCCA` — shared CC embedding (this release)
2. ⏳ `L2CCA` — provided as `cca_py.l2_normalize`; integration with the result struct pending
3. ⏳ `FindIntegrationAnchors` — k-NN in CCA space → mutual nearest neighbours → anchor scoring
4. ⏳ `IntegrateData` — anchor-weighted correction of the expression matrix

PRs welcome.

## Citation

If you use this package, please cite the original Seurat integration paper:

> Stuart, T., Butler, A., Hoffman, P., Hafemeister, C. *et al.* **Comprehensive integration of single-cell data.** *Cell* 177, 1888–1902 (2019). <https://doi.org/10.1016/j.cell.2019.05.031>

and acknowledge omicverse / this repo for the Python port.

## License

GNU GPLv3 — matches both upstream `omicverse` and Seurat.
