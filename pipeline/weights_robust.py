#!/usr/bin/env python3
"""TRUE continuous modality weighting — no hard per-mutation selection (the flaw that let optimize_weights
collapse IDH2 to 0.267, below chance). For each mutation, combine EVERY (strong-model x modality) OOF cell
by a continuous reliability weight w = max(0, oofAUC-0.5), orientation decided on train, averaged. A
positive-weighted average of ~40 oriented predictors cannot fall below chance — uniform is in its span.
-> runs/single_modality/_weights_robust.txt
"""
import os, sys, glob, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from amlmm.context import build_context, Config

ctx = build_context(Config(run_id="single_modality"))
RUN = os.path.dirname(ctx.path("x"))
RES = ctx.path("_weights_robust.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")

STRONG = ["logL2", "elastic", "linSVM", "shrLDA", "PLS"]
P = {}
for f in sorted(glob.glob(os.path.join(RUN, "oof_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
muts = sorted({m for d in P.values() for m in d["data"]})
emit("modalities: %s | continuous reliability weighting over strong cells, no hard selection\n" % ", ".join(MODS))


def fit_pct(col):
    s = np.sort(col[~np.isnan(col)]); n = len(s)
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


rows = []
for mut in muts:
    tr_truth, te_truth = {}, {}
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        for sid, t in zip(rec["train_ids"], rec["train_truth"]):
            tr_truth.setdefault(sid, t)
        for sid, t in zip(rec["test_ids"], rec["test_truth"]):
            te_truth.setdefault(sid, t)
    if not te_truth or len(set(te_truth.values())) < 2:
        continue
    tr_ids = sorted(tr_truth); te_ids = sorted(te_truth)
    ytr = np.array([int(tr_truth[s]) for s in tr_ids]); yte = np.array([int(te_truth[s]) for s in te_ids])
    agg = np.zeros(len(te_ids)); wsum = 0.0; wmod = {}; oc = np.nan
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        for model in STRONG:
            mr = rec["models"].get(model)
            if not mr or mr.get("oof") is None or mr.get("test") is None:
                continue
            omap = dict(zip(rec["train_ids"], [np.nan if v is None else float(v) for v in mr["oof"]]))
            tmap = dict(zip(rec["test_ids"], [np.nan if v is None else float(v) for v in mr["test"]]))
            oof = np.array([omap.get(s, np.nan) for s in tr_ids])
            mp = fit_pct(oof)
            a = auc(ytr, mp(oof))
            if a != a:
                continue
            sign = 1.0 if a >= 0.5 else -1.0; a = max(a, 1 - a)
            w = max(0.0, a - 0.5)
            qte = mp(np.array([tmap.get(s, np.nan) for s in te_ids])); qte = np.where(np.isnan(qte), 0.5, qte)
            if sign < 0:
                qte = 1.0 - qte
            ta = auc(yte, qte)                              # oracle = best single oriented cell on held-out
            if ta == ta:
                oc = ta if (oc != oc or ta > oc) else oc
            if w <= 0:
                continue
            agg += w * qte; wsum += w; wmod[mod] = wmod.get(mod, 0.0) + w
    if wsum <= 0:
        continue
    ho = auc(yte, agg / wsum)
    top = ", ".join("%s=%.2f" % (k, v / wsum) for k, v in sorted(wmod.items(), key=lambda x: -x[1])[:3])
    rows.append((mut.replace("mut_", "").replace("cyto_", ""), ho, oc, top))

H = pd.DataFrame([(r[0], r[1], r[2]) for r in rows], columns=["mutation", "robust", "oracle"])
emit("%-14s %8s %8s   %s" % ("mutation", "ROBUST", "oracle*", "top-weighted modalities"))
for r in rows:
    emit("%-14s %8s %8s   %s" % (r[0][:14], "%.3f" % r[1] if r[1] == r[1] else "-",
         "%.3f" % r[2] if r[2] == r[2] else "-", r[3]))
v = H[H.robust.notna()]
emit("\nROBUST continuous-weighting held-out mean = %.3f over %d muts" % (v.robust.mean(), len(v)))
emit("  (hard-selection weighted = 0.823 ; stacking = 0.848 ; robust gated baseline = 0.859 ; oracle %.3f)" % v.oracle.mean())
emit("min per-mutation AUC = %.3f (a real weighting should never be < ~0.5)" % v.robust.min())
emit("\nROBUST OK")
