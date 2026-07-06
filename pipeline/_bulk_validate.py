#!/usr/bin/env python3
"""Round-trip validation of the bulk deconvolution keystone (Phase D bulk path).
Run on a compute node:  PYTHONIOENCODING=utf-8 python _bulk_validate.py

Diagnostic version: NNLS against a CP10k signature recovers each state's COUNT-fraction (share
of total RNA in the mixture), which is NOT the same as the CELL-fraction (composition) because
states differ in RNA-per-cell. So we grade the recovered weights against BOTH the count-fraction
the synthetic bulk actually encodes (the math's own target) and the cell-fraction composition
(what the panel ultimately wants) — to separate "deconvolution broken" from "wrong yardstick".
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import amlmm
from amlmm import bulk
from amlmm.dataio import CELLSTATE_COL

ctx = amlmm.build_context(amlmm.Config(run_id="bulk_validate"))
a = bulk._load_rna(ctx)
print("RNA atlas loaded:", a.shape)
sig = bulk.cellstate_signature(ctx, adata=a)
print(f"signature: {sig.shape[0]} cell-states x {sig.shape[1]} marker genes\n")

genes = [str(g) for g in a.var_names]
obs_states = a.obs[CELLSTATE_COL].astype(str).values
obs_names = a.obs_names.astype(str)
pb = ctx.tables["pseudobulks"]
comp = ctx.tables["composition"]
cand = [s for s in comp.index if s in set(pb["sample_key"])][:40]
print(f"validating round-trip on {len(cand)} atlas samples\n")

def corr(u, v):
    if u.std() == 0 or v.std() == 0:
        return np.nan
    return float(np.corrcoef(u, v)[0, 1])

r_count, r_cell, top_count, top_cell, n = [], [], 0, 0, 0
for sk in cand:
    rows = set(pb.index[pb["sample_key"] == sk])
    mask = np.array([nm in rows for nm in obs_names])
    if not mask.any():
        continue
    Xs = a.X[mask]
    Xs = np.asarray(Xs.todense()) if hasattr(Xs, "todense") else np.asarray(Xs, dtype=float)
    bulkvec = pd.Series(Xs.sum(axis=0).ravel(), index=genes)
    rec = bulk.deconvolve(bulkvec, sig)
    # ground truth A: count-fraction (per-state total counts / sample total) = what NNLS targets
    st = obs_states[mask]
    perpb = Xs.sum(axis=1).ravel()
    cf = {}
    for s, t in zip(st, perpb):
        cf[s] = cf.get(s, 0.0) + float(t)
    countfrac = pd.Series(cf).reindex(sig.index).fillna(0.0)
    countfrac = countfrac / countfrac.sum() if countfrac.sum() else countfrac
    # ground truth B: cell-fraction composition (what the panel wants)
    cellfrac = comp.loc[sk].reindex(sig.index).fillna(0.0)
    cellfrac = cellfrac / cellfrac.sum() if cellfrac.sum() else cellfrac
    n += 1
    r_count.append(corr(rec.values, countfrac.values))
    r_cell.append(corr(rec.values, cellfrac.values))
    if rec.idxmax() == countfrac.idxmax():
        top_count += 1
    if rec.idxmax() == cellfrac.idxmax():
        top_cell += 1

print(f"n={n}")
print(f"  vs COUNT-fraction (the math's own target):  median r={np.nanmedian(r_count):.3f}, "
      f"dominant-state match={top_count}/{n}")
print(f"  vs CELL-fraction (panel composition):        median r={np.nanmedian(r_cell):.3f}, "
      f"dominant-state match={top_cell}/{n}")
print("\nReading: if COUNT-fraction r is high but CELL-fraction r is low -> the NNLS works, the gap "
      "is the RNA-per-cell scaling (fixable). If BOTH are low -> the 89 fine states are too "
      "collinear for NNLS and the path needs coarser states / a dedicated signature.")
