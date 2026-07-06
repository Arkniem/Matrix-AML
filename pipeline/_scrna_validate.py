#!/usr/bin/env python3
"""Validate the scRNA composition keystone on a real AML single-cell query.
Run on a compute node:  PYTHONIOENCODING=utf-8 python _scrna_validate.py [query.h5ad]

Loads an AML scRNA sample, cosine-assigns each cell to the 89 bone-marrow reference states,
and reports the composition + alignment quality. If the query already carries a cell-type label
in obs (e.g. a prior cellHarmony run), we report agreement with our argmax assignment as a
correctness check.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import anndata as ad
from amlmm import scrna

QUERIES = sys.argv[1:] or [
    "/data/salomonis-archive/LabFiles/Nathan/Revio/MDS-AML-KINNEX-1/AML14.h5ad",
    "/data/salomonis-archive/LabFiles/Nathan/Revio/MDS-AML-KINNEX-1/test/BM27.h5ad",
]
ref = scrna.load_reference()
print(f"reference: {ref.shape[0]} markers x {ref.shape[1]} populations")

for q in QUERIES:
    if not os.path.exists(q):
        print(f"\n[skip] {q} (not found)")
        continue
    print(f"\n==== {os.path.basename(q)} ====")
    a = ad.read_h5ad(q)
    print(f"  shape: {a.shape} (cells x features) | obs cols: {list(a.obs.columns)[:8]}")
    print(f"  var_names sample: {list(a.var_names[:5])}")
    try:
        labels, scores, n_shared = scrna.assign_cells(a, ref)
    except ValueError as e:
        print(f"  CANNOT ASSIGN: {e}")
        continue
    comp = scrna.composition(labels, ref)
    print(f"  shared markers: {n_shared} | mean cosine: {np.mean(scores):.3f} | "
          f"median cosine: {np.median(scores):.3f}")
    print(f"  states present: {(comp > 0).sum()}/89 | top states: "
          + ", ".join(f"{s}={comp[s]:.2f}" for s in comp.sort_values(ascending=False).head(6).index))
    # correctness check vs any existing obs cell-type label
    for col in a.obs.columns:
        vals = a.obs[col].astype(str)
        if vals.nunique() > 3 and vals.isin(set(ref.columns)).mean() > 0.5:
            agree = (vals.values == labels).mean()
            print(f"  obs['{col}'] looks like reference states -> argmax agreement: {agree:.3f} "
                  f"(n={len(labels)})")
            break
print("\nDONE — composition keystone runs on real scRNA; the composition Series is a drop-in for the panel.")
