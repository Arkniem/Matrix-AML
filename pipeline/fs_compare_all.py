#!/usr/bin/env python3
"""FS vs non, ALL modalities: per-modality mean held-out AUC under
   full features  vs  differential FS at 500 (250/side)  vs  100 (50/side).
Small modalities load direct; GRN/RNA/Cell-comm from the disk caches (no OOM). Fast linear models.
-> runs/single_modality/_fs_compare_all.txt
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
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_fs_compare_all.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
hold = set(ctx.holdout)

def load_mod(MOD):
    cache = ctx.path("_sl_%s.pkl" % MOD)
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    if MOD == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(ctx.tables["samples"].index))
    if MOD == "ADT":
        return dataio.sample_modality_matrix(ctx, "ADT")
    if MOD == "Lipid":
        return dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
    if MOD == "Metabolite":
        return dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
    if MOD == "LSC":
        t = ctx.tables.get("lsc_calls")
        cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")
    raise SystemExit("no loader/cache for %s" % MOD)

MODS = ["Composition", "RNA", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
SPECS = [("logL2", LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), "p"),
         ("linSVM", LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), "d"),
         ("PLS", None, "pls")]
def sc_pred(spec, Xtr, ytr, Xte):
    n, base, k = spec
    if k == "pls":
        return PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()
    e = clone(base).fit(Xtr, ytr)
    return e.decision_function(Xte) if k == "d" else e.predict_proba(Xte)[:, 1]

def diff_sel(Xtr, ytr, kside):
    if Xtr.shape[1] <= 2 * kside:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0])
    md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F)
    return np.unique(np.concatenate([o[:kside], o[-kside:]]))

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
FULL_CAP = 6000   # Cell-comm full is infeasible -> compare against a top-6000 proxy (noted)

rows = []
for MOD in MODS:
    try:
        B = load_mod(MOD).fillna(0.0)
        B = B[~B.index.duplicated(keep="first")]
    except Exception as e:
        emit("%-12s LOAD FAILED: %s" % (MOD, e)); continue
    nfeat = B.shape[1]
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(B.index); inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    acc = {"full": [], "fs500": [], "fs100": []}
    for m in MUTS:
        yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
        tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
        te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
        ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
        if len(set(yte)) < 2:
            continue
        X0 = B.loc[tr].values; Xt0 = B.loc[te].values
        keep = X0.std(0) > 0; X0, Xt0 = X0[:, keep], Xt0[:, keep]
        sc = StandardScaler().fit(X0); Xtr_a, Xte_a = sc.transform(X0), sc.transform(Xt0)
        if Xtr_a.shape[1] > FULL_CAP:                       # Cell-comm: full-proxy = top-6000 by |F|
            sub = diff_sel(Xtr_a, ytr, FULL_CAP // 2); Xtr_a, Xte_a = Xtr_a[:, sub], Xte_a[:, sub]
        s5 = diff_sel(Xtr_a, ytr, 250); s1 = diff_sel(Xtr_a, ytr, 50)
        for spec in SPECS:
            try: acc["full"].append(roc_auc_score(yte, sc_pred(spec, Xtr_a, ytr, Xte_a)))
            except Exception: pass
            try: acc["fs500"].append(roc_auc_score(yte, sc_pred(spec, Xtr_a[:, s5], ytr, Xte_a[:, s5])))
            except Exception: pass
            try: acc["fs100"].append(roc_auc_score(yte, sc_pred(spec, Xtr_a[:, s1], ytr, Xte_a[:, s1])))
            except Exception: pass
    fu, f5, f1 = np.mean(acc["full"]), np.mean(acc["fs500"]), np.mean(acc["fs100"])
    rows.append((MOD, nfeat, fu, f5, f1))
    emit("%-12s done (%d feat)" % (MOD, nfeat))

emit("\n%-12s %7s %8s %8s %8s   %9s %9s" % ("modality", "nfeat", "fullAUC", "FS-500", "FS-100", "500-full", "100-full"))
for MOD, nfeat, fu, f5, f1 in rows:
    note = " (full=top6000 proxy)" if nfeat > FULL_CAP else ("" if nfeat > 500 else "  [<=500: FS=full]")
    emit("%-12s %7d %8.3f %8.3f %8.3f   %+9.3f %+9.3f%s" % (MOD, nfeat, fu, f5, f1, f5 - fu, f1 - fu, note))
emit("\nFS COMPARE ALL OK")
