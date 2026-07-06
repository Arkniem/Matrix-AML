#!/usr/bin/env python3
"""Optimal per-(mutation, model) modality weights + gated deference + sealed held-out eval (audited).

Reads runs/single_modality/oof_<MOD>.pkl (honest donor-grouped out-of-fold TRAIN scores + held-out TEST
scores per modality/mutation/model). For each (mutation, model) it learns the optimal continuous
non-negative modality-weight vector via an INNER donor-grouped CV over the OOF rows; the held-out samples
are scored exactly once at the end.

Audited fixes vs v1 (all selection-inflation, no leakage was present):
  * percentile maps are FIT PER INNER FOLD on train rows and applied to val/test (no cross-split bleed);
  * each modality is ORIENTED by its train-fold AUC (flip if <0.5) and DROPPED if non-informative, so
    anti-correlated modalities never drag the blend with a small positive weight;
  * the 'single' (deference) strategy is honest CV: pick the best modality ON THE TRAIN FOLD, score it on
    val — not a max-over-columns on the val rows;
  * MODEL-AVG is scored by a REAL inner-CV AUC of the averaged predictor (per-fold averaged val preds),
    comparable to the per-model candidates;
  * deployment config chosen by a ONE-STANDARD-ERROR rule over a SMALL candidate set (defer<avg<blend by
    simplicity), with the inner-CV spread reported;
  * held-out MEAN and oracle MEAN share one nan-aware denominator; oracle uses NaN (no -1 sentinel);
  * sanity gate: if the held-out mean reaches the oracle, that is flagged as a selection-inflation signal.

Strategies: single (defer) | softmax w_j ∝ exp((trainAUC_j-0.5)/tau) shrunk to uniform by alpha | nnls
(ridge non-negative least squares of oriented train-quantiles onto truth). Hyper-params by inner CV.
-> runs/single_modality/_weights.txt + learned_weights.tsv + heldout_optimized.tsv
"""
import os, sys, glob, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from amlmm.context import build_context, Config

ctx = build_context(Config(run_id="single_modality"))
RUN = os.path.dirname(ctx.path("x"))
RES = ctx.path("_weights.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
dg = ctx.tables["samples"]["donor_group"].astype(str)

STRONG = ["logL2", "elastic", "linSVM", "shrLDA", "PLS"]
ALLM = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]
BLEND = [("softmax", (0.05, 0.25)), ("softmax", (0.05, 0.5)),
         ("softmax", (0.15, 0.25)), ("softmax", (0.15, 0.5)), ("nnls", 0.5), ("nnls", 2.0)]
FIXED_AVG = ("softmax", (0.15, 0.5))                       # low-DOF strategy used inside MODEL-AVG
EPS = 1e-9

P = {}
for f in sorted(glob.glob(os.path.join(RUN, "oof_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
muts = sorted({m for d in P.values() for m in d["data"]})
emit("modalities: %s" % ", ".join(MODS))
emit("strong pool: %s | inner CV = donor-grouped 3-fold | blend grid=%d configs\n" % (", ".join(STRONG), len(BLEND)))


def auc(y, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    if ok.sum() >= 4 and len(set(y[ok])) == 2:
        return roc_auc_score(y[ok], s[ok])
    return np.nan


def fit_pct(train_col):
    """quantile mapper fit on TRAIN values; maps any value -> its train-distribution quantile (NaN-safe)."""
    s = np.sort(train_col[~np.isnan(train_col)]); n = len(s)
    def f(x):
        x = np.asarray(x, float); out = np.full(len(x), np.nan); ok = ~np.isnan(x)
        if n >= 2:
            out[ok] = np.searchsorted(s, x[ok], side="right") / n
        elif n == 1:
            out[ok] = 0.5
        return out
    return f


def matrices(mut, model):
    tr_truth, te_truth, tr_raw, te_raw = {}, {}, {}, {}
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        mr = rec["models"].get(model)
        if not mr:
            continue
        for sid, t in zip(rec["train_ids"], rec["train_truth"]):
            tr_truth.setdefault(sid, t)
        for sid, t in zip(rec["test_ids"], rec["test_truth"]):
            te_truth.setdefault(sid, t)
        if mr.get("oof") is not None:
            tr_raw[mod] = dict(zip(rec["train_ids"], [np.nan if v is None else float(v) for v in mr["oof"]]))
        if mr.get("test") is not None:
            te_raw[mod] = dict(zip(rec["test_ids"], [np.nan if v is None else float(v) for v in mr["test"]]))
    mod_list = [m for m in MODS if m in tr_raw and m in te_raw]
    if not mod_list or not tr_truth or not te_truth:
        return None
    tr_ids = sorted(tr_truth); te_ids = sorted(te_truth)
    R = np.array([[tr_raw[m].get(s, np.nan) for m in mod_list] for s in tr_ids])
    Rte = np.array([[te_raw[m].get(s, np.nan) for m in mod_list] for s in te_ids])
    y = np.array([int(tr_truth[s]) for s in tr_ids]); yte = np.array([int(te_truth[s]) for s in te_ids])
    grp = np.array([dg.get(s, s) for s in tr_ids])
    return tr_ids, y, grp, R, te_ids, yte, Rte, mod_list


def fit_combiner(R, y, rows, strat, hp):
    """Fit percentile maps + orientation + weights on R[rows] only."""
    M = R.shape[1]; maps = []; sign = np.ones(M); aucs = np.full(M, 0.5); keep = np.zeros(M, bool)
    for j in range(M):
        mp = fit_pct(R[rows, j]); maps.append(mp)
        q = mp(R[rows, j]); a = auc(y[rows], q); a = 0.5 if a != a else a
        if a < 0.5:
            sign[j] = -1.0; a = 1.0 - a
        aucs[j] = a; keep[j] = a > 0.5 + EPS
    if not keep.any():
        keep[np.argmax(aucs)] = True
    if strat == "single":
        w = np.zeros(M); w[np.argmax(np.where(keep, aucs, -1))] = 1.0
    elif strat == "softmax":
        tau, al = hp
        z = np.where(keep, np.exp((aucs - 0.5) / tau), 0.0)
        z = z / z.sum() if z.sum() > 0 else keep / keep.sum()
        u = keep / keep.sum()
        w = (1 - al) * z + al * u
    else:  # nnls
        lam = hp
        cols = []
        for j in range(M):
            q = maps[j](R[rows, j]); q = q if sign[j] > 0 else 1 - q
            cols.append(np.where(np.isnan(q), 0.5, q))
        Q = np.column_stack(cols)
        A = np.vstack([Q, np.sqrt(lam) * np.eye(M)]); b = np.concatenate([y[rows].astype(float), np.zeros(M)])
        w, _ = nnls(A, b); w = w * keep
        if w.sum() <= 0:
            w = np.zeros(M); w[np.argmax(np.where(keep, aucs, -1))] = 1.0
    return {"maps": maps, "sign": sign, "w": w}


def predict_on(C, Rmat):
    M = Rmat.shape[1]; cols = []
    for j in range(M):
        q = C["maps"][j](Rmat[:, j]); cols.append(q if C["sign"][j] > 0 else 1 - q)
    Q = np.column_stack(cols); w = C["w"]; out = np.full(Rmat.shape[0], np.nan)
    for i in range(Rmat.shape[0]):
        ok = (~np.isnan(Q[i])) & (w > 0)               # only positively-weighted, available modalities
        if ok.any() and w[ok].sum() > 0:
            out[i] = float(np.dot(Q[i, ok], w[ok]) / w[ok].sum())
    return out


def inner_folds(grp, y):
    ng = len(set(grp))
    if ng < 2:
        return []
    return [(tri, vai) for tri, vai in GroupKFold(min(3, ng)).split(np.zeros(len(y)), y, grp)
            if len(set(y[tri])) == 2 and len(set(y[vai])) == 2]


def cv_eval(R, y, folds, strat, hp):
    """per-fold AUCs -> (mean, std) of an honestly fold-isolated combiner."""
    per = []
    for tri, vai in folds:
        C = fit_combiner(R, y, tri, strat, hp)
        a = auc(y[vai], predict_on(C, R[vai]))
        if a == a:
            per.append(a)
    return (np.mean(per), np.std(per)) if per else (np.nan, np.nan)


wrows, hrows = [], []
for mut in muts:
    mfit = {}
    for model in ALLM:
        mm = matrices(mut, model)
        if mm is None:
            continue
        tr_ids, y, grp, R, te_ids, yte, Rte, mod_list = mm
        folds = inner_folds(grp, y)
        if not folds:
            continue
        s_mean, s_std = cv_eval(R, y, folds, "single", None)
        blends = [(st, hp) + cv_eval(R, y, folds, st, hp) for st, hp in BLEND]
        bb = max(blends, key=lambda t: (t[2] if t[2] == t[2] else -1))   # (strat,hp,mean,std)
        sc_single = -1 if s_mean != s_mean else s_mean
        sc_blend = -1 if bb[2] != bb[2] else bb[2]
        chosen = ("single", None, s_mean, s_std) if sc_single >= sc_blend else (bb[0], bb[1], bb[2], bb[3])
        C_full = fit_combiner(R, y, np.arange(len(y)), chosen[0], chosen[1])
        wn = C_full["w"] / (C_full["w"].sum() or 1.0)
        mfit[model] = {"chosen": chosen, "C": C_full, "wn": wn, "mods": mod_list,
                       "R": R, "y": y, "folds": folds, "Rte": Rte, "yte": yte,
                       "single": (s_mean, s_std), "blend": bb}
        for j, mod in enumerate(mod_list):
            wrows.append((mut.replace("mut_", "").replace("cyto_", ""), model, chosen[0], mod,
                          round(float(wn[j]), 4), round(float(chosen[2]), 3) if chosen[2] == chosen[2] else None))
    if not mfit:
        continue
    pool = {m: mfit[m] for m in STRONG if m in mfit} or mfit
    yte = next(iter(pool.values()))["yte"]
    cands = []                                          # (name, mean, std, test_pred, dof_rank)
    cs = max(pool, key=lambda m: (pool[m]["single"][0] if pool[m]["single"][0] == pool[m]["single"][0] else -1))
    f = pool[cs]; Cs = fit_combiner(f["R"], f["y"], np.arange(len(f["y"])), "single", None)
    cands.append(("defer:" + cs, f["single"][0], f["single"][1], predict_on(Cs, f["Rte"]), 1))
    cb = max(pool, key=lambda m: (pool[m]["blend"][2] if pool[m]["blend"][2] == pool[m]["blend"][2] else -1))
    f = pool[cb]
    cands.append((("blend:%s:%s" % (cb, f["blend"][0])), f["blend"][2], f["blend"][3], predict_on(f["C"], f["Rte"]), 2))
    if len(pool) >= 2:                                  # MODEL-AVG: real inner-CV of the averaged predictor
        folds = next(iter(pool.values()))["folds"]; per = []
        for tri, vai in folds:
            preds = []
            for m, f in pool.items():
                C = fit_combiner(f["R"], f["y"], tri, FIXED_AVG[0], FIXED_AVG[1])
                preds.append(predict_on(C, f["R"][vai]))
            a = auc(f["y"][vai], np.nanmean(np.vstack(preds), axis=0))
            if a == a:
                per.append(a)
        avg_mean, avg_std = (np.mean(per), np.std(per)) if per else (np.nan, np.nan)
        tpreds = []
        for m, f in pool.items():
            C = fit_combiner(f["R"], f["y"], np.arange(len(f["y"])), FIXED_AVG[0], FIXED_AVG[1])
            tpreds.append(predict_on(C, f["Rte"]))
        cands.append(("MODEL-AVG", avg_mean, avg_std, np.nanmean(np.vstack(tpreds), axis=0), 0))
    cands = [c for c in cands if c[1] == c[1]]
    if not cands:
        continue
    bestm = max(c[1] for c in cands); bstd = [c[2] for c in cands if c[1] == bestm][0]
    se = (bstd / np.sqrt(3)) if bstd == bstd else 0.0
    within = [c for c in cands if c[1] >= bestm - se]
    dep = min(within, key=lambda c: (c[4], -c[1]))      # one-SE: simplest config within a SE of the best
    ho = auc(yte, dep[3])
    oc = np.nan
    for m, f in mfit.items():
        for j in range(f["Rte"].shape[1]):
            q = fit_pct(f["R"][:, j])(f["Rte"][:, j]); a = auc(f["yte"], q)
            if a == a:
                a = max(a, 1 - a); oc = a if (oc != oc or a > oc) else oc
    hrows.append((mut.replace("mut_", "").replace("cyto_", ""), dep[0],
                  round(float(dep[1]), 3), round(float(dep[2]), 3) if dep[2] == dep[2] else None,
                  round(float(ho), 3) if ho == ho else np.nan, round(float(oc), 3) if oc == oc else np.nan))

W = pd.DataFrame(wrows, columns=["mutation", "model", "strategy", "modality", "weight", "innerCV_auc"])
W.to_csv(os.path.join(RUN, "learned_weights.tsv"), sep="\t", index=False)
H = pd.DataFrame(hrows, columns=["mutation", "deployed", "innerCV", "innerSD", "heldout_auc", "oracle"])
H.to_csv(os.path.join(RUN, "heldout_optimized.tsv"), sep="\t", index=False)

emit("%-14s %-18s %8s %7s %9s %8s" % ("mutation", "deployed", "innerCV", "±SD", "HELDOUT", "oracle*"))
for r in hrows:
    emit("%-14s %-18s %8.3f %7s %9s %8s" % (
        r[0][:14], r[1][:18], r[2], ("%.2f" % r[3]) if r[3] is not None else "-",
        ("%.3f" % r[4]) if r[4] == r[4] else "-", ("%.3f" % r[5]) if r[5] == r[5] else "-"))
valid = H[H.heldout_auc.notna()]
hmean = valid.heldout_auc.mean(); omean = valid.oracle.mean()
emit("%-14s %-18s %8s %7s %9.3f %8.3f" % ("MEAN", "(valid muts)", "", "", hmean, omean))
defer = int(H.deployed.str.startswith("defer").sum()); avg = int((H.deployed == "MODEL-AVG").sum())
emit("\noptimized held-out mean AUC = %.3f  (prior best 0.859 ; oracle ceiling %.3f)" % (hmean, omean))
emit("deployment: gated-deference %d, MODEL-AVG %d, blend %d  of %d mutations"
     % (defer, avg, len(valid) - defer - avg, len(valid)))
emit("learned_weights.tsv: %d (mutation,model,modality) optimal weights" % len(W))
if hmean >= omean - EPS:
    emit("\n*** SANITY WARNING: held-out mean >= oracle — possible selection inflation, investigate ***")
emit("\nWEIGHTS OK")
