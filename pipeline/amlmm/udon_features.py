"""UDON RNA representations for the predictor (the lab's preferred RNA encoding).

Two sample-level blocks, both crosswalking each per-pseudobulk row -> sample_key via (Sample, cell_state)
[verified 9668/9668 udon rows map cleanly] and aggregating over the sample's cell-states:

  udon_fold_sample_matrix  -> sample x gene  control-normalized FOLD vectors (vs healthy baseline;
                              udon_result.h5ad, values are ratios ~1.0, n_cells-weighted mean, log1p)
  udon_program_matrix      -> sample x UDON-program membership FRACTIONS (RNA_UDON_final_program_assignments)

Coverage: UDON folds cover ~337/387 samples; samples without folds are simply absent (caller handles).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from . import pseudobulk_io as pio, dataio
from .dataio import CELLSTATE_COL


def _sample_xwalk(ctx):
    """(Sample, cell_state) -> sample_key and -> n_cells, from the pipeline pseudobulks table."""
    pb = ctx.tables["pseudobulks"]
    sk, nc = {}, {}
    for s, c, k, n in zip(pb["sample"].astype(str), pb["cell_state"].astype(str),
                          pb["sample_key"].astype(str), pb["n_cells"].astype(float)):
        sk[(s, c)] = k
        nc[(s, c)] = n
    return sk, nc


def udon_fold_sample_matrix(ctx, markers=True, log=True) -> pd.DataFrame:
    """Sample x gene control-normalized UDON folds; n_cells-weighted mean over the sample's cell-states."""
    mk = pio.udon_markers(ctx, "RNA") if markers else None
    X, obs = pio.udon_result_matrix(ctx, "RNA", markers=mk)
    if X is None or X.shape[0] == 0:
        return pd.DataFrame(index=[])
    cs_col = "cell_state" if "cell_state" in obs.columns else CELLSTATE_COL
    sk, nc = _sample_xwalk(ctx)
    S = obs["Sample"].astype(str).values
    C = obs[cs_col].astype(str).values
    keys = [sk.get((s, c)) for s, c in zip(S, C)]
    w = np.array([nc.get((s, c), 1.0) for s, c in zip(S, C)], dtype=float)
    df = X.copy()
    df["_k"] = keys
    df["_w"] = np.clip(w, 1.0, None)
    df = df[df["_k"].notna()]

    def wm(b):
        ww = b["_w"].values
        v = b.drop(columns=["_k", "_w"], errors="ignore")
        return pd.Series(np.average(v.values, axis=0, weights=ww), index=v.columns)

    out = df.groupby("_k", sort=False).apply(wm)
    return np.log1p(out.clip(lower=0)) if log else out


def canonical_rna(ctx) -> pd.DataFrame:
    """The empirically-best RNA block = log1p raw expression on UDON markers + UDON program-membership
    fractions. (vs raw alone on donor-grouped CV: +0.026 FLT3, +0.021 IDH2, neutral NPM1/TP53/DNMT3A;
    the control-normalized FOLD vectors were tested and HURT mutation prediction, so we keep raw+programs.)
    Reuses the _sl_RNA.pkl aggregation cache when present (avoids the heavy full-RNA re-aggregation)."""
    cache = ctx.path("_sl_RNA.pkl")
    if os.path.exists(cache):
        r = pd.read_pickle(cache)
    else:
        r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0))
        mk = pio.udon_markers(ctx, "RNA")
        if mk:
            r = r[[g for g in mk if g in r.columns]]
    prog = udon_program_matrix(ctx)
    if len(prog):
        r = r.join(prog, how="left")
    return r.fillna(0.0)


def udon_program_matrix(ctx) -> pd.DataFrame:
    """Sample x UDON-program membership fractions (fraction of the sample's pseudobulks in each program)."""
    P = ctx.tables.get("udon_programs")
    if P is None or not len(P):
        return pd.DataFrame(index=[])
    P = P.copy()
    sk, _ = _sample_xwalk(ctx)
    S = P["Sample"].astype(str).values
    pid = P["pseudobulk"].astype(str).values
    cs = [p[:-(len(s) + 2)] if p.endswith("__" + s) else p.split("__", 1)[0] for p, s in zip(pid, S)]
    P["_k"] = [sk.get((s, c)) for s, c in zip(S, cs)]
    P = P[P["_k"].notna()]
    if not len(P):
        return pd.DataFrame(index=[])
    ct = P.groupby(["_k", P["final_program"].astype(str)]).size().unstack(fill_value=0)
    frac = ct.div(ct.sum(axis=1).clip(lower=1), axis=0)
    frac.columns = ["UDONprog_%s" % c for c in frac.columns]
    return frac
