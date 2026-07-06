#!/usr/bin/env python3
"""Stacking meta-learner — the richest combiner.

For each mutation, treat every (strong-model x modality) OOF prediction as a base-learner column and fit a
regularized logistic META-learner over the whole library jointly (vs optimize_weights, which weights
modalities within a single model). Combines across BOTH models and modalities.

Leakage-clean, reusing the audited optimize_weights patterns:
  * per-cell percentile map FIT on the fold's train rows, applied to val/test (no cross-split bleed);
  * meta-learner regularization C chosen by INNER donor-grouped CV on the OOF rows;
  * the sealed held-out is scored exactly once.
Overfitting guard: 40 correlated cells on modest positives -> strong-L2 grid biased low + inner-CV C +
a sanity gate (stack >= oracle flags selection inflation).
-> runs/single_modality/_stack.txt + stack_heldout.tsv
"""
import os, sys, glob, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from amlmm.context import build_context, Config

ctx = build_context(Config(run_id="single_modality"))
RUN = os.path.dirname(ctx.path("x"))
RES = ctx.path("_stack.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
dg = ctx.tables["samples"]["donor_group"].astype(str)

STRONG = ["logL2", "elastic", "linSVM", "shrLDA", "PLS"]
CGRID = [0.01, 0.02, 0.05, 0.1, 0.3]                       # strong L2 first (40 correlated cells, few positives)

P = {}
for f in sorted(glob.glob(os.path.join(RUN, "oof_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
muts = sorted({m for d in P.values() for m in d["data"]})
emit("modalities: %s | meta over up to %d (strong-model x modality) cells\n" % (", ".join(MODS), len(STRONG) * len(MODS)))


def fit_pct(col):
    s = np.sort(col[~np.isnan(col)]); n = len(s)
    def f(x):
        x = np.asarray(x, float); out = np.full(len(x), np.nan); ok = ~np.isnan(x)
        if n >= 2:
            out[ok] = np.searchsorted(s, x[ok], side="right") / n
        elif n == 1:
            out[ok] = 0.5
        return out
    return f


def design(M, maps):
    cols = []
    for j in range(M.shape[1]):
        q = maps[j](M[:, j]); cols.append(np.where(np.isnan(q), 0.5, q))   # impute missing/unscored -> neutral
    return np.column_stack(cols)


def auc(y, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if (ok.sum() >= 4 and len(set(y[ok])) == 2) else np.nan


def inner_folds(grp, y):
    ng = len(set(grp))
    if ng < 2:
        return []
    return [(tri, vai) for tri, vai in GroupKFold(min(3, ng)).split(np.zeros(len(y)), y, grp)
            if len(set(y[tri])) == 2 and len(set(y[vai])) == 2]


def cells(mut):
    tr_truth, te_truth, tr_cols, te_cols = {}, {}, {}, {}
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        for model in STRONG:
            mr = rec["models"].get(model)
            if not mr or mr.get("oof") is None or mr.get("test") is None:
                continue
            cn = "%s:%s" % (model, mod)
            tr_cols[cn] = dict(zip(rec["train_ids"], [np.nan if v is None else float(v) for v in mr["oof"]]))
            te_cols[cn] = dict(zip(rec["test_ids"], [np.nan if v is None else float(v) for v in mr["test"]]))
        for sid, t in zip(rec["train_ids"], rec["train_truth"]):
            tr_truth.setdefault(sid, t)
        for sid, t in zip(rec["test_ids"], rec["test_truth"]):
            te_truth.setdefault(sid, t)
    if not tr_cols or not te_truth:
        return None
    cn = sorted(tr_cols)
    tr_ids = sorted(tr_truth); te_ids = sorted(te_truth)
    OOF = np.array([[tr_cols[c].get(s, np.nan) for c in cn] for s in tr_ids])
    TEST = np.array([[te_cols[c].get(s, np.nan) for c in cn] for s in te_ids])
    y = np.array([int(tr_truth[s]) for s in tr_ids]); y_te = np.array([int(te_truth[s]) for s in te_ids])
    grp = np.array([dg.get(s, s) for s in tr_ids])
    return tr_ids, y, grp, OOF, te_ids, y_te, TEST, cn


def meta_fit(X, y, C):
    return LogisticRegression(C=C, penalty="l2", class_weight="balanced", max_iter=4000).fit(X, y)


rows = []
for mut in muts:
    c = cells(mut)
    if c is None:
        continue
    tr_ids, y, grp, OOF, te_ids, y_te, TEST, cn = c
    folds = inner_folds(grp, y)
    if not folds or len(set(y_te)) < 2:
        continue
    bestC, bestA = None, -1.0
    for C in CGRID:                                        # inner-CV select regularization (refit maps per fold)
        ps = np.full(len(y), np.nan)
        for tri, vai in folds:
            maps = [fit_pct(OOF[tri][:, j]) for j in range(OOF.shape[1])]
            try:
                ps[vai] = meta_fit(design(OOF[tri], maps), y[tri], C).predict_proba(design(OOF[vai], maps))[:, 1]
            except Exception:
                pass
        a = auc(y, ps)
        if a == a and a > bestA:
            bestA, bestC = a, C
    if bestC is None:
        continue
    maps = [fit_pct(OOF[:, j]) for j in range(OOF.shape[1])]     # refit on full train, score held-out once
    try:
        ho = auc(y_te, meta_fit(design(OOF, maps), y, bestC).predict_proba(design(TEST, maps))[:, 1])
    except Exception:
        ho = np.nan
    oc = np.nanmax([auc(y_te, fit_pct(OOF[:, j])(TEST[:, j])) for j in range(OOF.shape[1])] + [np.nan])
    rows.append((mut.replace("mut_", "").replace("cyto_", ""), len(cn), bestC, round(float(bestA), 3),
                 round(float(ho), 3) if ho == ho else None, round(float(oc), 3) if oc == oc else None))

H = pd.DataFrame(rows, columns=["mutation", "ncells", "C", "innerCV", "heldout_stack", "oracle"])
H.to_csv(os.path.join(RUN, "stack_heldout.tsv"), sep="\t", index=False)
emit("%-14s %6s %5s %8s %10s %8s" % ("mutation", "cells", "C", "innerCV", "STACK", "oracle*"))
for r in rows:
    emit("%-14s %6d %5.2f %8.3f %10s %8s" % (r[0][:14], r[1], r[2], r[3],
         ("%.3f" % r[4]) if r[4] is not None else "-", ("%.3f" % r[5]) if r[5] is not None else "-"))
valid = H[H.heldout_stack.notna()]
hm = valid.heldout_stack.mean(); om = valid.oracle.mean()
emit("\nSTACK held-out mean = %.3f over %d muts  (weighted-fusion was 0.859 ; oracle %.3f)" % (hm, len(valid), om))
if hm >= om - 1e-9:
    emit("*** SANITY: stack >= oracle — investigate selection inflation ***")
emit("\nSTACK OK")
