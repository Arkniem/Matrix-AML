#!/usr/bin/env python3
"""Full per-(mutation, model) view: optimized modality weights vs UNIFORM (un-optimized), all 8 modalities.

For each mutation and each strong model, that model predicts every modality (8 oriented OOF/test columns).
  uniform   = equal-weight mean of all available modalities (the 'without optimized weights' baseline)
  optimized = ridge-NNLS modality weights solved on OOF (all 8 in the pool; some get 0 = they don't help)
Reports: mutation x model held-out AUC for optimized and for uniform, per-model means, and the full
8-modality weight vector per mutation (averaged across the strong models). -> _weights_full.txt
"""
import os, sys, glob, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score
from amlmm.context import build_context, Config

ctx = build_context(Config(run_id="single_modality"))
RUN = os.path.dirname(ctx.path("x"))
RES = ctx.path("_weights_full.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
STRONG = ["logL2", "elastic", "linSVM", "shrLDA", "PLS"]
P = {}
for f in sorted(glob.glob(os.path.join(RUN, "oof_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())                                      # all 8 modalities
muts = sorted({m for d in P.values() for m in d["data"]})


def fit_pct(c):
    s = np.sort(c[~np.isnan(c)]); n = len(s)
    def f(x):
        x = np.asarray(x, float); o = np.full(len(x), np.nan); ok = ~np.isnan(x)
        if n >= 2:
            o[ok] = np.searchsorted(s, x[ok], side="right") / n
        elif n == 1:
            o[ok] = 0.5
        return o
    return f


def auc(y, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if (ok.sum() >= 4 and len(set(y[ok])) == 2) else np.nan


def model_cols(mut, model, tr, te, ytr):
    names, Ctr, Cte = [], [], []
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        mr = rec["models"].get(model)
        if not mr or mr.get("oof") is None or mr.get("test") is None:
            continue
        omap = dict(zip(rec["train_ids"], [np.nan if v is None else float(v) for v in mr["oof"]]))
        tmap = dict(zip(rec["test_ids"], [np.nan if v is None else float(v) for v in mr["test"]]))
        oof = np.array([omap.get(s, np.nan) for s in tr]); mp = fit_pct(oof); a = auc(ytr, mp(oof))
        if a != a:
            continue
        sign = 1.0 if a >= 0.5 else -1.0
        qtr = np.where(np.isnan(mp(oof)), 0.5, mp(oof))
        qte = mp(np.array([tmap.get(s, np.nan) for s in te])); qte = np.where(np.isnan(qte), 0.5, qte)
        if sign < 0:
            qtr, qte = 1 - qtr, 1 - qte
        names.append(mod); Ctr.append(qtr); Cte.append(qte)
    return names, Ctr, Cte


OPT = {m: {} for m in muts}; UNI = {m: {} for m in muts}; WACC = {m: {} for m in muts}
for mut in muts:
    tr_truth, te_truth = {}, {}
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        for s, t in zip(rec["train_ids"], rec["train_truth"]):
            tr_truth.setdefault(s, t)
        for s, t in zip(rec["test_ids"], rec["test_truth"]):
            te_truth.setdefault(s, t)
    if not te_truth or len(set(te_truth.values())) < 2:
        continue
    tr = sorted(tr_truth); te = sorted(te_truth)
    ytr = np.array([int(tr_truth[s]) for s in tr]); yte = np.array([int(te_truth[s]) for s in te])
    for model in STRONG:
        names, Ctr, Cte = model_cols(mut, model, tr, te, ytr)
        if len(names) < 2:
            continue
        Otr = np.column_stack(Ctr); Ote = np.column_stack(Cte)
        UNI[mut][model] = auc(yte, Ote.mean(1))                    # uniform: equal weights, all modalities
        A = np.vstack([Otr, np.eye(Otr.shape[1])]); b = np.concatenate([ytr.astype(float), np.zeros(Otr.shape[1])])
        w, _ = nnls(A, b)
        if w.sum() <= 0:
            w = np.ones(len(names))
        OPT[mut][model] = auc(yte, Ote @ w / w.sum())
        for nm, wi in zip(names, w / w.sum()):
            WACC[mut].setdefault(nm, []).append(wi)


def tbl(D, title):
    emit("\n=== %s : held-out AUC (mutation x model) ===" % title)
    emit("%-14s %s   %6s" % ("mutation", " ".join("%-7s" % m for m in STRONG), "mean"))
    for mut in muts:
        if not D[mut]:
            continue
        vals = [D[mut].get(m, np.nan) for m in STRONG]
        emit("%-14s %s   %6.3f" % (mut.replace("mut_", "").replace("cyto_", "")[:14],
             " ".join(("%.3f" % v if v == v else "  -  ").ljust(7) for v in vals), np.nanmean(vals)))
    pm = [np.nanmean([D[mt].get(m, np.nan) for mt in muts if D[mt]]) for m in STRONG]
    allv = np.nanmean([v for mt in muts for v in D[mt].values()])
    emit("%-14s %s   %6.3f" % ("MEAN", " ".join("%.3f".ljust(7) % p for p in pm), allv))


tbl(OPT, "OPTIMIZED modality weights")
tbl(UNI, "UNIFORM (no optimized weights)")

emit("\n=== per-model mean held-out AUC: optimized vs uniform ===")
emit("%-9s %10s %10s %8s" % ("model", "optimized", "uniform", "delta"))
for m in STRONG:
    o = np.nanmean([OPT[mt][m] for mt in muts if m in OPT[mt]])
    u = np.nanmean([UNI[mt][m] for mt in muts if m in UNI[mt]])
    emit("%-9s %10.3f %10.3f %+8.3f" % (m, o, u, o - u))
oall = np.nanmean([v for mt in muts for v in OPT[mt].values()])
uall = np.nanmean([v for mt in muts for v in UNI[mt].values()])
emit("%-9s %10.3f %10.3f %+8.3f" % ("ALL", oall, uall, oall - uall))

emit("\n=== molded modality weights — ALL 8 modalities (mean across strong models) ===")
emit("%-14s %s" % ("mutation", " ".join("%-6s" % m[:6] for m in MODS)))
for mut in muts:
    if not WACC[mut]:
        continue
    ws = [np.mean(WACC[mut].get(m, [0.0])) for m in MODS]
    ws = [w / (sum(ws) or 1) for w in ws]
    emit("%-14s %s" % (mut.replace("mut_", "").replace("cyto_", "")[:14], " ".join("%5.2f " % w for w in ws)))
emit("\nFULL OK")
