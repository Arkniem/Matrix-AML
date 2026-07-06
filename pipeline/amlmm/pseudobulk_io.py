"""Per-pseudobulk I/O for the Discovery agent (and the Deploy NN seam).

The modality .h5ads are ALREADY per-pseudobulk (12,255 rows, one per sample x cell-state); `dataio`
deliberately AGGREGATES them to one vector per sample. Discovery needs the opposite: the raw
per-pseudobulk vectors, sliced to one cell-state at a time (within a cell-state there is exactly one
pseudobulk per sample, so a per-(modality x cell-state) matrix is a sample-level matrix on a row subset
-> `cv.nested_cv_evaluate` applies unchanged with donor grouping).

Also home to the UDON signature machinery (the user-confirmed matching method): the per-modality
`<MOD>/clusters/udon_result.h5ad` is the control-normalized fold/signature matrix; the MarkerFinder
feature list (`marker_heatmap.txt` col 1, or RNA's `UDON_final_program_markers.txt`) restricts the
features; matching is **Spearman rank correlation** restricted to those markers + the same cell-state.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import dataio
from .dataio import IMPUTED_MODALITIES, CELLSTATE_COL, feature_fidelity, _key


# --------------------------------------------------------------------------- per-pseudobulk modality
def _mem_modality(ctx, modality):
    """Modality AnnData loaded NON-backed and cached on ctx. Backed SPARSE reads are broken on the
    current cluster stack (anndata 0.10.8 + scipy 1.15 / numpy 2.4 raise in the backed CSR
    reconstruction); a one-time full load + in-memory slicing is correct and fast (read once, slice
    every cell-state). Same workaround dataio.cellcomm_matrix already uses for its file."""
    key = f"_mem::{modality}"
    a = ctx.tables.get(key)
    if a is None:
        import anndata as ad
        a = ad.read_h5ad(ctx._modality_paths[modality])
        ctx.tables[key] = a
    return a


def pseudobulk_modality_matrix(ctx, modality, pseudobulk_ids=None, min_spearman=None) -> pd.DataFrame:
    """Per-pseudobulk feature matrix (rows = pseudobulk ids, cols = features) — NO sample aggregation.
    Like dataio.sample_modality_matrix but skips the groupby wmean. Imputed modalities are
    fidelity-filtered to features with held-out Spearman >= min_spearman when set."""
    if modality not in ctx._modality_paths:
        return pd.DataFrame(index=[])
    pb = ctx.tables["pseudobulks"]
    a = _mem_modality(ctx, modality)
    ids = list(pseudobulk_ids) if pseudobulk_ids is not None else list(pb.index)
    pos = {name: i for i, name in enumerate(a.obs_names.astype(str))}
    rows = [r for r in ids if r in pos]
    if not rows:
        return pd.DataFrame(index=[])
    idx = np.asarray([pos[r] for r in rows], dtype=np.intp)
    X = a.X[idx]                                            # in-memory row slice (sparse or dense)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    out = pd.DataFrame(X, index=rows, columns=list(a.var_names.astype(str)))
    if min_spearman is not None and modality in IMPUTED_MODALITIES:
        fid = feature_fidelity(ctx, modality)
        if fid is not None:
            keep = set(fid[fid >= min_spearman].index)
            out = out[[c for c in out.columns if c in keep]]
    return out


def cellstate_pseudobulks(ctx, cell_state, sample_keys=None) -> list:
    """Pseudobulk ids of one cell-state (optionally restricted to sample_keys)."""
    pb = ctx.tables["pseudobulks"]
    m = pb["cell_state"].astype(str) == str(cell_state)
    if sample_keys is not None:
        m &= pb["sample_key"].isin(set(sample_keys))
    return list(pb.index[m])


def cellstate_modality_matrix(ctx, modality, cell_state, sample_keys=None, min_spearman=None) -> pd.DataFrame:
    """Per-pseudobulk modality matrix for ONE cell-state (one row per sample with that state).
    Rows indexed by pseudobulk id; join to ctx.tables['pseudobulks'] for sample_key / donor / label."""
    ids = cellstate_pseudobulks(ctx, cell_state, sample_keys)
    if not ids:
        return pd.DataFrame(index=[])
    return pseudobulk_modality_matrix(ctx, modality, pseudobulk_ids=ids, min_spearman=min_spearman)


def cell_state_sizes(ctx) -> pd.Series:
    """n pseudobulks per cell-state (population), descending — for tractable iteration order."""
    pb = ctx.tables["pseudobulks"]
    return pb["cell_state"].astype(str).value_counts()


# --------------------------------------------------------------------------- RNA normalization (fold-safe)
def cp10k_log1p(counts) -> np.ndarray:
    """CP10k + log1p over a (rows x genes) count matrix (matches scrna.assign_cells / rna2* bundles)."""
    X = np.asarray(counts, dtype=np.float64)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(X / tot * 1e4).astype(np.float32)


def make_normalizer(kind="none"):
    """A fold-safe sklearn FunctionTransformer (applied INSIDE the CV Pipeline, no leakage).
    kind: 'cp10k_log1p' (RNA raw counts) | 'none' (already-normalized imputed modalities)."""
    from sklearn.preprocessing import FunctionTransformer
    if kind == "cp10k_log1p":
        return FunctionTransformer(cp10k_log1p, feature_names_out="one-to-one")
    return FunctionTransformer(feature_names_out="one-to-one")


# --------------------------------------------------------------------------- UDON signature machinery
_MARKER_META = {"", "row_clusters-flat", "column_clusters-flat", "marker", "uid", "uniqueid", "genes"}


def marker_finder_features(path) -> list:
    """Feature list from a MarkerFinder file: first column, skipping header/cluster-annotation rows.
    Handles both formats: AltAnalyze heatmap (`marker_heatmap.txt`: col1 empty / 'column_clusters-flat'
    header rows then features) and RNA's tidy `UDON_final_program_markers.txt` ('marker' header then genes)."""
    feats, seen = [], set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            c0 = line.split("\t", 1)[0].strip()
            if c0.lower() in _MARKER_META:
                continue
            if c0 not in seen:
                seen.add(c0)
                feats.append(c0)
    return feats


def _clusters_dir(ctx, modality) -> str:
    return os.path.join(ctx.config.base_dir, modality, "clusters")


def udon_marker_path(ctx, modality) -> "str | None":
    d = _clusters_dir(ctx, modality)
    for name in ("marker_heatmap.txt", "UDON_final_program_markers.txt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def udon_result_path(ctx, modality) -> "str | None":
    p = os.path.join(_clusters_dir(ctx, modality), "udon_result.h5ad")
    return p if os.path.exists(p) else None


def udon_markers(ctx, modality) -> list:
    p = udon_marker_path(ctx, modality)
    return marker_finder_features(p) if p else []


def _mem_udon(ctx, modality):
    """udon_result.h5ad loaded NON-backed + cached (same backed-stack workaround as _mem_modality)."""
    key = f"_udonmem::{modality}"
    a = ctx.tables.get(key)
    if a is None:
        path = udon_result_path(ctx, modality)
        if path is None:
            return None
        import anndata as ad
        a = ad.read_h5ad(path)
        ctx.tables[key] = a
    return a


def udon_result_matrix(ctx, modality, markers=None, cell_state=None):
    """The control-normalized UDON fold/signature matrix `<MOD>/clusters/udon_result.h5ad`, restricted
    to `markers` (intersection with var) and optionally one `cell_state`. Returns (X: DataFrame
    [pseudobulk_id x marker], obs: DataFrame with Sample/cell_state/Annotation), or (None, None) if the
    file is absent. obs has Sample + cell_state + Annotation but NO Dataset (crosswalk to sample_key is
    on (sample, cell_state), disambiguated by Annotation)."""
    a = _mem_udon(ctx, modality)
    if a is None:
        return None, None
    obs = a.obs.copy()
    obs.index = a.obs_names.astype(str)
    cs_col = CELLSTATE_COL if CELLSTATE_COL in obs.columns else (
        "cell_state" if "cell_state" in obs.columns else None)
    row_mask = np.ones(a.n_obs, dtype=bool)
    if cell_state is not None and cs_col is not None:
        row_mask = (obs[cs_col].astype(str) == str(cell_state)).values
    ridx = np.where(row_mask)[0].astype(np.intp)
    if len(ridx) == 0:
        return pd.DataFrame(index=[]), obs.iloc[[]]
    Xfull = a.X[ridx]                                      # in-memory row slice
    Xfull = np.asarray(Xfull.todense()) if hasattr(Xfull, "todense") else np.asarray(Xfull)
    var = list(a.var_names.astype(str))
    ridn = list(obs.index[ridx])
    if markers is not None:
        vpos = {g: i for i, g in enumerate(var)}
        cols = [vpos[g] for g in markers if g in vpos]
        if not cols:
            return pd.DataFrame(index=ridn), obs.iloc[ridx]
        X = Xfull[:, cols]
        cnames = [var[c] for c in cols]
    else:
        X = Xfull
        cnames = var
    Xdf = pd.DataFrame(X, index=ridn, columns=cnames)
    o = obs.iloc[ridx].copy()
    o.index = ridn
    if cs_col and cs_col != "cell_state":
        o = o.rename(columns={cs_col: "cell_state"})
    return Xdf, o


def _spearman_to_each(query, ref) -> np.ndarray:
    """Spearman rank correlation of a query vector vs each row of ref (aligned columns).
    Spearman(a,b) = Pearson(rank(a), rank(b)); rank along the feature axis, then standardized dot."""
    from scipy.stats import rankdata
    q = rankdata(np.asarray(query, dtype=float))
    R = rankdata(np.asarray(ref, dtype=float), axis=1)
    qc = q - q.mean()
    qn = qc / (np.linalg.norm(qc) or 1.0)
    Rc = R - R.mean(axis=1, keepdims=True)
    Rn = Rc / (np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12)
    return Rn @ qn


def udon_signature_match(query, ref_matrix, k=5):
    """Top-k nearest training pseudobulks by Spearman rank correlation (the confirmed UDON-match method).
    query: 1D array over markers. ref_matrix: DataFrame [pseudobulk_id x markers] aligned to the SAME
    marker columns (caller restricts to shared markers + the same cell-state). Returns a DataFrame
    [neighbor pseudobulk_id, spearman] sorted desc, top-k."""
    if ref_matrix is None or ref_matrix.shape[0] == 0:
        return pd.DataFrame(columns=["neighbor", "spearman"])
    corr = _spearman_to_each(query, ref_matrix.values)
    order = np.argsort(-corr)[:k]
    return pd.DataFrame({"neighbor": [ref_matrix.index[i] for i in order],
                         "spearman": [float(corr[i]) for i in order]})
