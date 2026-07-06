#!/usr/bin/env python3
"""Understand udon_result.h5ad so we can aggregate the control-normalized RNA folds to sample level:
   does obs_names match pipeline pseudobulk ids? how to crosswalk to sample_key? are values folds (neg)?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, anndata as ad
from amlmm.context import build_context, Config
from amlmm import pseudobulk_io as pio
from amlmm.dataio import CELLSTATE_COL, _key

ctx = build_context(Config(run_id="diag"))
pb = ctx.tables["pseudobulks"]
p = pio.udon_result_path(ctx, "RNA")
a = ad.read_h5ad(p, backed="r")
on = a.obs_names.astype(str)
print("udon shape:", a.shape, "| obs cols:", list(a.obs.columns))
print("udon obs_names[:3]:", list(on[:3]))
print("pseudobulks index[:3]:", list(pb.index.astype(str)[:3]))
inter = set(on) & set(pb.index.astype(str))
print("obs_name INTERSECT pseudobulk_id: %d / %d udon rows" % (len(inter), a.n_obs))

# crosswalk via (Sample, cell_state) if obs_names don't match
cs_col = CELLSTATE_COL if CELLSTATE_COL in a.obs.columns else a.obs.columns[1]
print("udon Sample[:3]:", list(a.obs["Sample"].astype(str)[:3]))
print("udon cell_state col '%s'[:3]:" % cs_col, list(a.obs[cs_col].astype(str)[:3]))
print("udon Annotation[:3]:", list(a.obs["Annotation"].astype(str)[:3]))
# does pb have n_cells / a (Sample,cell_state)->sample_key map?
print("pb cols:", list(pb.columns))
udf = pd.DataFrame({"Sample": a.obs["Sample"].astype(str).values, "cs": a.obs[cs_col].astype(str).values}, index=on)
pbkey = pb.assign(cs=pb["cell_state"].astype(str)).set_index(["sample", "cs"])["sample_key"]
udf["sk"] = [pbkey.get((s, c)) for s, c in zip(udf["Sample"], udf["cs"])]
print("udon rows crosswalked to a sample_key via (Sample,cell_state): %d / %d"
      % (int(udf["sk"].notna().sum()), len(udf)))
print("distinct sample_keys covered by udon folds:", udf["sk"].dropna().nunique())

X = a.X[:100]; X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
print("X[:100] min=%.3f max=%.3f mean=%.4f frac_neg=%.3f (folds should have negatives)"
      % (float(X.min()), float(X.max()), float(X.mean()), float((X < 0).mean())))
mk = pio.udon_markers(ctx, "RNA")
print("UDON markers:", len(mk), "| in udon var:", len(set(mk) & set(a.var_names.astype(str))))
print("DIAG OK")
