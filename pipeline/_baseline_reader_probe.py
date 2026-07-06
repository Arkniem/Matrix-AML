#!/usr/bin/env python3
"""Confirm the baseline dataio.sample_modality_matrix non-backed fix works on the upgraded stack
(backed-sparse reads previously crashed). Reads a dense (ADT) and a sparse (Metabolite) modality and
cross-checks one sample's aggregated vector against the per-pseudobulk reader (must be equivalent)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from amlmm.context import build_context, Config
from amlmm import dataio, pseudobulk_io as pio

ctx = build_context(Config(run_id="baseline_reader_probe"))
for mod in ("ADT", "Metabolite", "RNA"):
    if mod not in ctx._modality_paths:
        print(f"  {mod}: (not deposited, skip)"); continue
    M = dataio.sample_modality_matrix(ctx, mod)
    print(f"  {mod}: sample_modality_matrix -> {M.shape} (no crash)")
    assert M.shape[0] > 0 and M.shape[1] > 0, f"{mod} empty"

# equivalence cross-check on ADT: a single-cell-state sample's aggregated vector == its one pseudobulk
pb = ctx.tables["pseudobulks"]
counts = pb.groupby("sample_key")["cell_state"].nunique()
single = counts[counts == 1]
if len(single):
    sk = single.index[0]
    agg = dataio.sample_modality_matrix(ctx, "ADT", sample_keys=[sk])
    pbid = list(pb.index[pb["sample_key"] == sk])
    raw = pio.pseudobulk_modality_matrix(ctx, "ADT", pseudobulk_ids=pbid)
    if len(agg) and len(raw):
        cols = [c for c in agg.columns if c in raw.columns]
        close = np.allclose(agg.iloc[0][cols].values.astype(float),
                            raw.iloc[0][cols].values.astype(float), rtol=1e-4, atol=1e-4)
        print(f"  single-cell-state sample {sk}: agg==pseudobulk -> {close}")
        assert close, "aggregation diverges from the per-pseudobulk vector on a single-state sample"

print("BASELINE READER PROBE OK")
