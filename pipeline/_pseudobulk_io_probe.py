#!/usr/bin/env python3
"""Probe pseudobulk_io.py against the live atlas: per-pseudobulk cell-state reader (one row per sample),
MarkerFinder parsing per modality, udon_result.h5ad reader, and the Spearman self-match sanity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amlmm.context import build_context
from amlmm import pseudobulk_io as pio

ctx = build_context()
pb = ctx.tables["pseudobulks"]
print("layout", ctx.layout, "| pseudobulks", pb.shape, "| samples", ctx.tables["samples"].shape)
sizes = pio.cell_state_sizes(ctx)
print("top cell-states:", [(s, int(n)) for s, n in sizes.head(5).items()])
C = sizes.index[0]

print(f"\n== per-pseudobulk reader @ cell-state {C!r} (n_pb={int(sizes.iloc[0])}) ==")
for mod in ["ADT", "RNA", "GRN", "Metabolite", "Lipid"]:
    ms = 0.3 if mod in ("Metabolite", "Lipid") else None
    M = pio.cellstate_modality_matrix(ctx, mod, C, min_spearman=ms)
    sk = pb.loc[M.index, "sample_key"] if M.shape[0] else []
    print(f"  {mod:11s} shape={M.shape}  unique_samples={len(set(sk))}  "
          f"dup_per_sample={M.shape[0]-len(set(sk))}  idx_in_pb={all(i in pb.index for i in M.index)}")

print("\n== UDON readers (markers + fold object) ==")
for mod in ["RNA", "GRN", "ADT", "Lipid", "Metabolite"]:
    mk = pio.udon_markers(ctx, mod)
    rp = pio.udon_result_path(ctx, mod)
    print(f"  {mod:11s} markers={len(mk):4d}  udon_result={'yes' if rp else 'NO '}  e.g.={mk[:3]}")

print("\n== Spearman self-match sanity (RNA udon_result, marker-restricted, one cell-state) ==")
mk = pio.udon_markers(ctx, "RNA")
X, obs = pio.udon_result_matrix(ctx, "RNA", markers=mk, cell_state=C)
if X is not None and X.shape[0] >= 3:
    print(f"  matrix @ {C}: {X.shape}  shared_markers={X.shape[1]}  obs_cols={list(obs.columns)}")
    nn = pio.udon_signature_match(X.iloc[0].values, X, k=3)
    print("  top-3 for row0 (row0 must be itself, spearman~1.0):")
    print(nn.to_string(index=False))
    if "Annotation" in obs.columns:
        print("  neighbor annotations:", list(obs.loc[nn["neighbor"], "Annotation"]))
else:
    print("  [udon_result unavailable or too few rows here]")
print("\nPROBE OK")
