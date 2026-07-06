#!/usr/bin/env python3
"""One-off: do controls exist in the cohort, how are they labeled, and what is the UDON fold matrix?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, anndata as ad
from amlmm.context import build_context, Config
from amlmm import genetics, pseudobulk_io as pio

ctx = build_context(Config(run_id="diag"))
s = ctx.tables["samples"]
print("N samples total:", len(s))
dc = s.get("disease_category")
print("--- disease_category value_counts ---")
print(dc.fillna("NA").value_counts().head(25) if dc is not None else "NO disease_category COLUMN")
M = genetics.build_mutation_matrix(ctx)
flags = [c for c in M.columns if c.startswith(("mut_", "cyto_"))]
print("--- all-sample mutation labels --- present(1)=%d absent(0)=%d NaN=%d"
      % (int((M[flags] == 1).sum().sum()), int((M[flags] == 0).sum().sum()), int(M[flags].isna().sum().sum())))
dcl = dc.astype(str).str.lower() if dc is not None else pd.Series("", index=s.index)
ctrl = dcl.str.contains("control|normal|healthy|non-leuk|benign|\\bhd\\b", regex=True, na=False)
print("control-ish samples:", int(ctrl.sum()))
if int(ctrl.sum()):
    cm = M.loc[ctrl[ctrl].index, flags]
    print("  control mutation labels: present=%d absent=%d NaN=%d"
          % (int((cm == 1).sum().sum()), int((cm == 0).sum().sum()), int(cm.isna().sum().sum())))
    print("  example control keys:", list(ctrl[ctrl].index[:5]))
    print("  their disease_category:", list(dc.loc[ctrl[ctrl].index].dropna().unique()[:6]))
p = pio.udon_result_path(ctx, "RNA")
print("--- udon_result.h5ad:", p)
if p:
    a = ad.read_h5ad(p, backed="r")
    print("  shape", a.shape, "| obs cols", list(a.obs.columns)[:10])
    print("  var head", list(a.var_names[:6]), "| n_var", a.n_vars)
mk = pio.udon_markers(ctx, "RNA")
print("UDON RNA markers available:", len(mk), "| programs table rows:", len(ctx.tables.get("udon_programs", [])))
print("DIAG OK")
