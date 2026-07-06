#!/usr/bin/env python3
"""GRN: differential FS (top-500 up + top-500 down) vs NO feature selection (full 7486), per model.
Loads the cached sample-level matrix (no OOM). Fast-on-high-dim models only. -> prints a table.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from amlmm.context import build_context, Config
from amlmm import discovery as D, genetics

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_compare_fs.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")

hold = set(ctx.holdout)
B = pd.read_pickle(ctx.path("_sl_GRN.pkl")).fillna(0.0)
emit("GRN  differential-FS (top-500 up + top-500 down) vs NO-FS (full %d features)\n" % B.shape[1])

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
         ("PLS", None, "pls"),
         ("RF", RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=0, n_jobs=2), "p"),
         ("NaiveB", GaussianNB(), "p"),
         ("kNN", KNeighborsClassifier(n_neighbors=15), "p")]
def sc_pred(spec, Xtr, ytr, Xte):
    n, base, k = spec
    if k == "pls":
        return PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()
    e = clone(base).fit(Xtr, ytr)
    return e.decision_function(Xte) if k == "d" else e.predict_proba(Xte)[:, 1]

agg = {s[0]: {"full": [], "diff": []} for s in SPECS}
for m in MUTS:
    yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
    tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2:
        continue
    X0 = B.loc[tr].values; Xt0 = B.loc[te].values
    keep = X0.std(0) > 0; X0, Xt0 = X0[:, keep], Xt0[:, keep]
    sc = StandardScaler().fit(X0); Xtr_f, Xte_f = sc.transform(X0), sc.transform(Xt0)
    F = np.nan_to_num(f_classif(Xtr_f, ytr)[0])
    md = Xtr_f[ytr == 1].mean(0) - Xtr_f[ytr == 0].mean(0)
    order = np.argsort(np.sign(md) * F)
    sel = np.unique(np.concatenate([order[:500], order[-500:]]))
    Xtr_d, Xte_d = Xtr_f[:, sel], Xte_f[:, sel]
    for spec in SPECS:
        try: agg[spec[0]]["full"].append(roc_auc_score(yte, sc_pred(spec, Xtr_f, ytr, Xte_f)))
        except Exception: pass
        try: agg[spec[0]]["diff"].append(roc_auc_score(yte, sc_pred(spec, Xtr_d, ytr, Xte_d)))
        except Exception: pass
    emit("  %s done" % m[:14])

emit("\n%-9s %10s %10s %10s" % ("model", "no-FS(full)", "differential", "diff-full"))
for s in SPECS:
    fu = np.mean(agg[s[0]]["full"]); di = np.mean(agg[s[0]]["diff"])
    emit("%-9s %10.3f %10.3f %+10.3f" % (s[0], fu, di, di - fu))
emit("\nmean over models: no-FS %.3f  differential %.3f"
     % (np.mean([np.mean(agg[s[0]]["full"]) for s in SPECS]),
        np.mean([np.mean(agg[s[0]]["diff"]) for s in SPECS])))
emit("\nCOMPARE FS OK")
