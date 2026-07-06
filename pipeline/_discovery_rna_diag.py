#!/usr/bin/env python3
"""Isolate the RNA-combo CPU hog: time nested_cv for logreg-only vs rf-only vs both."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from amlmm.context import build_context
from amlmm import discovery as D, cv, targets

ctx = build_context()
base = D.DiscoveryConfig(prefilter_features=2000, max_features=1500)
labels = D.labels_for_field(ctx, "subtype").dropna()
labels = labels[labels.isin(targets.usable_classes(labels, base.min_class_n))]
X, skv, donor, cohort = D._combo_frame(ctx, "RNA", "MultiLin-GMP-1", set(labels.index))
y = pd.Series([labels[k] for k in skv], index=X.index).astype(str)
dpc = pd.DataFrame({"y": y.values, "g": donor}).groupby("y")["g"].nunique()
keep = set(dpc[dpc >= base.min_donors_per_class].index)
mask = y.isin(keep).values
X, y = X.loc[mask], y[mask]; donor = np.asarray(donor)[mask]; cohort = np.asarray(cohort)[mask]
X = X[list(X.var(axis=0).sort_values(ascending=False).index[:base.prefilter_features])]
print(f"X={X.shape}  classes={sorted(set(y))}  n_donors={len(set(donor))}")

for tag, mods, perms in [("logreg perms5", ("logreg",), 5), ("rf perms5", ("rf",), 5),
                         ("rf+logreg perms5", ("rf", "logreg"), 5),
                         ("rf+logreg perms0", ("rf", "logreg"), 0)]:
    cfg = D.DiscoveryConfig(prefilter_features=2000, max_features=1500, models=mods)
    facto = D._factories("RNA", cfg, X.shape[1])
    t = time.time()
    res = cv.nested_cv_evaluate(X, y, donor, cohort, strategy="donor_kfold",
                                model_factories=facto, outer_splits=5, inner_splits=3,
                                n_permutations=perms)
    print(f"  {tag:18s}: ba={res.get('balanced_accuracy')} in {time.time()-t:.1f}s")
print("DIAG DONE")
