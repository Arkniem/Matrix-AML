"""Phase D (bulk path, increment 1) — reference-based deconvolution of a bulk RNA-seq profile
into cell-state composition, the first input the per-patient panel needs for a NEW sample.

Why this exists: the altanalyze3 checkout has no turn-key bulk deconvolver (cellHarmony is
single-cell label-transfer), and the rna2* imputation models operate on per-cell-state
pseudobulks. So a bulk sample needs its cell-state FRACTIONS estimated. Standard approach:
a cell-state x gene signature built from the RNA atlas + non-negative least squares (NNLS).
Swap in CIBERSORTx / a curated signature later if preferred — `deconvolve` just needs a
signature DataFrame.

Validated by round-trip (`_bulk_validate.py`): sum an atlas sample's per-state pseudobulks into
a synthetic bulk, deconvolve, and confirm the recovered fractions track the known composition.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .dataio import CELLSTATE_COL


def _cp10k(counts):
    """Linear counts-per-10k normalization (rows = obs, cols = genes)."""
    counts = np.asarray(counts, dtype=float)
    tot = counts.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return counts / tot * 1e4


def _load_rna(ctx):
    """Full (non-backed) RNA atlas load. Non-backed avoids the cluster anndata backed-CSR
    fancy-index segfault and is read once; the matrix is sparse counts."""
    cached = ctx.tables.get("_rna_adata")
    if cached is not None:
        return cached
    import anndata as ad
    a = ad.read_h5ad(ctx._modality_paths["RNA"])
    ctx.tables["_rna_adata"] = a
    return a


def cellstate_signature(ctx, min_pseudobulks=5, n_marker_genes=2000, adata=None):
    """Cell-state x gene CP10k signature from the RNA atlas pseudobulks (mean per state),
    restricted to the most cross-state-variable genes for a well-conditioned NNLS. Cached."""
    cached = ctx.tables.get("_cellstate_signature")
    if cached is not None:
        return cached
    a = adata if adata is not None else _load_rna(ctx)
    states = a.obs[CELLSTATE_COL].astype(str).values
    genes = [str(g) for g in a.var_names]
    X = a.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    cp = _cp10k(X).astype(np.float32)            # CP10k, float32 to halve peak memory
    del X
    # per-state mean via boolean masks (avoids materializing a 12k x 36k DataFrame)
    rows, kept = [], []
    for s in sorted(set(states)):
        m = (states == s)
        if m.sum() < min_pseudobulks:
            continue
        rows.append(cp[m].mean(axis=0))
        kept.append(s)
    sig = pd.DataFrame(np.vstack(rows), index=kept, columns=genes)
    markers = list(sig.var(axis=0).sort_values(ascending=False).head(n_marker_genes).index)
    sig = sig[markers]
    ctx.tables["_cellstate_signature"] = sig
    return sig


def deconvolve(bulk, signature):
    """bulk: Series gene->expression (raw counts or CP10k). signature: DataFrame state x gene.
    Returns Series state->fraction via NNLS on the shared genes (CP10k-scaled), summing to 1."""
    from scipy.optimize import nnls
    genes = [g for g in signature.columns if g in bulk.index]
    if not genes:
        return pd.Series(0.0, index=signature.index)
    S = signature[genes].to_numpy(dtype=float).T          # genes x states
    b = bulk.reindex(genes).fillna(0.0).to_numpy(dtype=float)
    if b.sum() > 0:
        b = b / b.sum() * 1e4                              # match the signature's CP10k scale
    x, _ = nnls(S, b)
    total = x.sum()
    if total == 0:
        return pd.Series(0.0, index=signature.index)
    return pd.Series(x / total, index=signature.index)


def synthetic_bulk(ctx, sample_key, adata=None):
    """Sum an atlas sample's per-cell-state RNA pseudobulks into a whole-sample 'bulk' profile
    (gene->summed counts) — used to round-trip-validate the deconvolution."""
    a = adata if adata is not None else _load_rna(ctx)
    pb = ctx.tables["pseudobulks"]
    rows = set(pb.index[pb["sample_key"] == sample_key])
    obs_names = a.obs_names.astype(str)
    mask = np.array([n in rows for n in obs_names])
    if not mask.any():
        return None
    X = a.X[mask]
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    return pd.Series(np.asarray(X).sum(axis=0).ravel(), index=[str(g) for g in a.var_names])


def deconvolve_to_composition(ctx, bulk, signature=None):
    """Public entry: bulk Series gene->counts -> cell-state fraction Series (a drop-in for the
    panel's `composition` row for a new sample)."""
    sig = signature if signature is not None else cellstate_signature(ctx)
    return deconvolve(bulk, sig)
