"""AnnData adapter — run CCA across two AnnData batches and write the
shared embedding back into ``adata.obsm['X_cca']``."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .cca import run_cca


def run_cca_anndata(
    adata1,
    adata2,
    features: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    num_cc: int = 20,
    standardize_inputs: bool = True,
    key_added: str = "X_cca",
    seed: Optional[int] = 42,
):
    """Run CCA on two AnnData objects and stash the embedding back.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Two **cell × gene** matrices to integrate. Genes will be aligned
        on ``var_names``; if ``features`` is given, only that subset is
        used (intersected with what's present in both).
    features : sequence[str] or None
        Explicit feature subset. Default is the intersection of both
        objects' variable-feature flags (``adata.var['highly_variable']``
        if present), or the full intersection of ``var_names``.
    layer : str or None
        If given, pull expression from ``adata.layers[layer]`` instead
        of ``adata.X``. Standard pattern: log-normalised counts.
    num_cc : int
        Number of canonical components.
    standardize_inputs : bool
        Z-score each gene per dataset before CCA.
    key_added : str
        Where to write the embedding. Each AnnData gets its half written
        to ``adata.obsm[key_added]`` (shape ``(n_obs, num_cc)``).
    seed : int or None
        Reproducibility seed.

    Returns
    -------
    cca_py.RunCCAResult
    """
    # 1. resolve features (gene intersection)
    var1 = set(adata1.var_names)
    var2 = set(adata2.var_names)
    common = var1 & var2
    if features is not None:
        common = common & set(features)
    elif "highly_variable" in adata1.var.columns and "highly_variable" in adata2.var.columns:
        hv1 = set(adata1.var_names[adata1.var["highly_variable"]])
        hv2 = set(adata2.var_names[adata2.var["highly_variable"]])
        common = common & (hv1 | hv2)
    common = sorted(common)
    if len(common) < num_cc:
        raise ValueError(
            f"Only {len(common)} shared features after filtering; "
            f"need at least num_cc={num_cc}."
        )

    # 2. extract feature × cell matrices
    def _matrix(ad):
        sub = ad[:, common]
        M = sub.layers[layer] if layer is not None else sub.X
        if hasattr(M, "toarray"):
            M = M.toarray()
        return np.asarray(M, dtype=np.float64).T  # AnnData is cells × genes → transpose

    X = _matrix(adata1)
    Y = _matrix(adata2)

    # 3. run CCA
    result = run_cca(
        X, Y,
        num_cc=num_cc,
        standardize_inputs=standardize_inputs,
        seed=seed,
    )

    # 4. write back per-dataset embeddings
    u, v = result.split()
    adata1.obsm[key_added] = u
    adata2.obsm[key_added] = v
    adata1.uns.setdefault("cca", {})[key_added] = {
        "num_cc": num_cc, "n_features": len(common), "singular_values": result.d.tolist(),
        "paired_with": adata2.shape[0],
    }
    adata2.uns.setdefault("cca", {})[key_added] = {
        "num_cc": num_cc, "n_features": len(common), "singular_values": result.d.tolist(),
        "paired_with": adata1.shape[0],
    }
    return result
