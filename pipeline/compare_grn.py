#!/usr/bin/env python3
"""GRN: full-features-for-8-models (new) vs top-500-all (baseline), per-model mean held-out AUC."""
import os, pandas as pd, numpy as np
RUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "single_modality")
new = pd.read_csv(os.path.join(RUN, "auc_GRN.tsv"), sep="\t")
old = pd.read_csv(os.path.join(RUN, "grn_top500_baseline.tsv"), sep="\t")
MODELS = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]
FS_NEW = {"elastic", "MLP", "shrLDA"}        # these stayed top-500 in the new run; the other 8 are full 7486
print("GRN  full-features(8 models) vs top-500(all)  -- per-model mean held-out AUC")
print("%-9s %9s %9s %9s   %s" % ("model", "new", "top500", "delta", "features(new)"))
for mo in MODELS:
    n = new[new.model == mo].auc.mean(); o = old[old.model == mo].auc.mean()
    feat = "top-500" if mo in FS_NEW else "full 7486"
    print("%-9s %9.3f %9.3f %+9.3f   %s" % (mo, n, o, n - o, feat))
