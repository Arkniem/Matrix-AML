#!/usr/bin/env python3
"""How high can per-mutation model choice + optimized modality weights go?

For each (mutation, model): optimized modality weights (ridge-NNLS on OOF), record OOF AUC + held-out AUC.
  ORACLE (UPPER BOUND, selection-on-test): per mutation pick the model with the best HELD-OUT AUC.
                                            NOT deployable — it peeks at the answer.
  DEPLOYABLE (leakage-clean):               per mutation pick the model with the best OOF AUC, report its
                                            held-out (model chosen on train only).
  FIXED:                                    the best single model used for every mutation (linSVM).
-> runs/single_modality/_best_model.txt
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
RES = ctx.path("_best_model.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
STRONG = ["logL2", "elastic", "linSVM", "shrLDA", "PLS"]
P = {}
for f in sorted(glob.glob(os.path.join(RUN, "oof_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
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
    Ctr, Cte = [], []
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
        Ctr.append(qtr); Cte.append(qte)
    return Ctr, Cte


rows = []
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
    pm = {}
    for model in STRONG:
        Ctr, Cte = model_cols(mut, model, tr, te, ytr)
        if len(Ctr) < 2:
            continue
        Otr = np.column_stack(Ctr); Ote = np.column_stack(Cte)
        A = np.vstack([Otr, np.eye(Otr.shape[1])]); b = np.concatenate([ytr.astype(float), np.zeros(Otr.shape[1])])
        w, _ = nnls(A, b)
        if w.sum() <= 0:
            w = np.ones(Otr.shape[1])
        pm[model] = (auc(ytr, Otr @ w / w.sum()), auc(yte, Ote @ w / w.sum()))
    if not pm:
        continue
    om = max(pm, key=lambda m: pm[m][1] if pm[m][1] == pm[m][1] else -1)        # ORACLE: best held-out
    dm = max(pm, key=lambda m: pm[m][0] if pm[m][0] == pm[m][0] else -1)        # DEPLOYABLE: best OOF
    lin = pm.get("linSVM", (np.nan, np.nan))[1]
    rows.append((mut.replace("mut_", "").replace("cyto_", ""), om, pm[om][1], dm, pm[dm][1], lin))

H = pd.DataFrame(rows, columns=["mut", "orc_model", "oracle", "dep_model", "deployable", "linSVM"])
emit("%-14s %-9s %8s   %-9s %10s   %8s" % ("mutation", "orc_mdl", "ORACLE*", "dep_mdl", "DEPLOYABLE", "linSVM"))
for r in rows:
    emit("%-14s %-9s %8.3f   %-9s %10.3f   %8s" % (r[0][:14], r[1], r[2], r[3], r[4],
         "%.3f" % r[5] if r[5] == r[5] else "-"))
emit("%-14s %-9s %8.3f   %-9s %10.3f   %8.3f" % ("MEAN", "", H.oracle.mean(), "", H.deployable.mean(), H.linSVM.mean()))
emit("\nORACLE (best model per mutation by HELD-OUT) = %.3f  <- UPPER BOUND, selection-on-test, NOT deployable"
     % H.oracle.mean())
emit("DEPLOYABLE (best model per mutation by OOF)  = %.3f  <- leakage-clean, what you can actually ship" % H.deployable.mean())
emit("FIXED best model (linSVM, every mutation)    = %.3f" % H.linSVM.mean())
emit("\nBEST MODEL OK")
