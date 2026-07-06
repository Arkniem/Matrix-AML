#!/usr/bin/env python3
"""Aggregate ONE modality (env AMLMM_MODALITY) to its sample-level matrix and pickle it.
No model fitting -> minimal memory, exits immediately -> won't OOM. The main run then loads
runs/single_modality/_sl_<MOD>.pkl instead of re-aggregating the multi-GB AnnData.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio

MOD = os.environ.get("AMLMM_MODALITY", "GRN")
ctx = build_context(Config(run_id="single_modality"))
hold = set(ctx.holdout)

if MOD == "GRN":
    B = dataio.sample_modality_matrix(ctx, "GRN")
elif MOD == "RNA":
    r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0))
    mk = pio.udon_markers(ctx, "RNA")
    B = r[[g for g in mk if g in r.columns]] if mk else r[list(
        r.loc[[s for s in r.index if s not in hold]].var().sort_values(ascending=False).head(2000).index)]
elif MOD == "Cell-comm":
    B = dataio.cellcomm_matrix(ctx)
elif MOD == "Lipid":
    B = dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
elif MOD == "Metabolite":
    B = dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
elif MOD == "ADT":
    B = dataio.sample_modality_matrix(ctx, "ADT")
elif MOD == "Composition":
    B = D._sample_level_matrix(ctx, "composition", set(ctx.tables["samples"].index))
elif MOD == "LSC":
    t = ctx.tables.get("lsc_calls")
    cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if t is not None and c in t.columns]
    B = t[cols].apply(pd.to_numeric, errors="coerce")
else:
    raise SystemExit("unknown modality %s" % MOD)

B = B[~B.index.duplicated(keep="first")]
out = ctx.path("_sl_%s.pkl" % MOD)
B.to_pickle(out)
print("cached %s: %s -> %s" % (MOD, B.shape, out))
