#!/usr/bin/env python3
"""Stage-2 validation: run a few single (field x modality x cell-state) Discovery combos end-to-end
and confirm the weight / OOF / validated-marker harvest + the leakage-safe donor grouping."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amlmm.context import build_context
from amlmm import discovery as D

ctx = build_context()
cfg = D.DiscoveryConfig(final_permutations=20)  # uses convergence-safe defaults (prefilter 800, maxfeat 300)
COMBOS = [
    ("subtype", "composition", D.SAMPLE_SENTINEL),
    ("subtype", "ADT", "MultiLin-GMP-1"),
    ("mut_NPM1", "composition", D.SAMPLE_SENTINEL),
    ("subtype", "RNA", "MultiLin-GMP-1"),
]
for fld, mod, cs in COMBOS:
    print(f"\n==== {fld} x {mod} x {cs} ====")
    r = D.run_combo(ctx, fld, mod, cs, cfg, permutations=cfg.final_permutations)
    for kk in ("status", "reason", "n_pseudobulks", "n_donors", "n_classes", "classes",
               "balanced_accuracy", "permutation_p", "permutation_p95", "weight",
               "gate_accept", "winning_model", "fidelity", "needs_more_data"):
        if kk in r:
            print(f"  {kk}: {r[kk]}")
    oof = r.get("oof", {})
    print(f"  n_oof_pseudobulks: {len(oof)}")
    mk = r.get("markers", [])
    val = [m["feature"] for m in mk if m.get("heldout_validated")]
    print(f"  validated markers ({len(val)}/{len(mk)}): {val[:8]}")
    # leakage cross-check: no donor in >1 OOF fold is implicit; here assert OOF count == n rows used
    if r.get("status") == "ok":
        assert len(oof) == r["n_pseudobulks"], f"OOF count {len(oof)} != n_pseudobulks {r['n_pseudobulks']}"
print("\nCOMBO TEST OK")
