#!/usr/bin/env python3
"""Settle whether FS-500 hurts GRN: held-out AUC, GRN whole (7486) vs top-500, fast models only.
-> runs/single_modality/_grn_fs_check.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, genetics

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_grn_fs_check.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    open(RES, "a", encoding="utf-8").write(str(m) + "\n")

samples = ctx.tables["samples"]; hold = set(ctx.holdout)
B = dataio.sample_modality_matrix(ctx, "GRN")
B = B[~B.index.duplicated(keep="first")].fillna(0.0)
emit("GRN %d features | comparing whole vs top-500\n" % B.shape[1])
M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
MUTS = []
for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
    y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(B.index); inh = y.index.isin(hold)
    if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
       and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
        MUTS.append(f)

SPECS = [("logL2", LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), "p"),
         ("linSVM", LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), "d"),
         ("shrLDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "p"),
         ("PLS", None, "pls"),
         ("RF", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=0, n_jobs=4), "p")]
def sc_pred(spec, Xtr, ytr, Xte):
    n, base, k = spec
    if k == "pls":
        return PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()
    e = clone(base).fit(Xtr, ytr)
    return e.decision_function(Xte) if k == "d" else e.predict_proba(Xte)[:, 1]

agg = {s[0]: {"whole": [], "top500": []} for s in SPECS}
for m in MUTS:
    yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
    tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2:
        continue
    X0 = B.loc[tr].values; Xt0 = B.loc[te].values
    keep = X0.std(0) > 0; X0, Xt0 = X0[:, keep], Xt0[:, keep]
    sc = StandardScaler().fit(X0); Xtr_w, Xte_w = sc.transform(X0), sc.transform(Xt0)
    skb = SelectKBest(f_classif, k=min(500, Xtr_w.shape[1])).fit(Xtr_w, ytr)
    Xtr_5, Xte_5 = skb.transform(Xtr_w), skb.transform(Xte_w)
    for spec in SPECS:
        try: agg[spec[0]]["whole"].append(roc_auc_score(yte, sc_pred(spec, Xtr_w, ytr, Xte_w)))
        except Exception: pass
        try: agg[spec[0]]["top500"].append(roc_auc_score(yte, sc_pred(spec, Xtr_5, ytr, Xte_5)))
        except Exception: pass
    emit("  %s done" % m[:14])

emit("\n%-9s %10s %10s %9s" % ("model", "whole7486", "top500", "FS-whole"))
for s in SPECS:
    w = np.mean(agg[s[0]]["whole"]); t = np.mean(agg[s[0]]["top500"])
    emit("%-9s %10.3f %10.3f %+9.3f" % (s[0], w, t, t - w))
emit("\nGRN FS CHECK OK")
