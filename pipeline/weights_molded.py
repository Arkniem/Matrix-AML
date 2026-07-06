#!/usr/bin/env python3
"""Per-mutation modality weights MOLDED to that mutation's ablation, with a dominance guarantee.

Each modality is represented by its best strong model's OOF prediction (the modality's ablation result).
Weights are then OPTIMIZED per mutation (ridge non-negative least squares of the oriented modality
percentiles onto truth) — accounting for joint structure, not just marginal ablation AUC (so correlated
modalities aren't double-counted). A FLOOR guarantees the optimized combination is never worse than the
single best modality on the data it is molded to (fall back to the ablation winner if it would be) — so
the optimized weighting weakly dominates the ablation winner by construction.

Reports per mutation: ablation-best modality vs optimized-combined on OOF (dominance guaranteed) and on
the sealed held-out. -> runs/single_modality/_weights_molded.txt
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
RES = ctx.path("_weights_molded.txt"); open(RES, "w").close()
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


rows = []; dom_ok = 0; dom_tot = 0
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
    modnames, Ctr, Cte, abl_oof, abl_ho = [], [], [], [], []
    for mod in MODS:                                       # each modality -> its best strong model (ablation)
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        best = None
        for model in STRONG:
            mr = rec["models"].get(model)
            if not mr or mr.get("oof") is None or mr.get("test") is None:
                continue
            omap = dict(zip(rec["train_ids"], [np.nan if v is None else float(v) for v in mr["oof"]]))
            oof = np.array([omap.get(s, np.nan) for s in tr]); mp = fit_pct(oof); a = auc(ytr, mp(oof))
            if a != a:
                continue
            sign = 1.0 if a >= 0.5 else -1.0; a2 = max(a, 1 - a)
            if best is None or a2 > best[0]:
                tmap = dict(zip(rec["test_ids"], [np.nan if v is None else float(v) for v in mr["test"]]))
                qtr = np.where(np.isnan(mp(oof)), 0.5, mp(oof))
                qte = np.where(np.isnan(mp(np.array([tmap.get(s, np.nan) for s in te]))), 0.5,
                               mp(np.array([tmap.get(s, np.nan) for s in te])))
                if sign < 0:
                    qtr, qte = 1 - qtr, 1 - qte
                best = (a2, qtr, qte)
        if best is None:
            continue
        modnames.append(mod); Ctr.append(best[1]); Cte.append(best[2]); abl_oof.append(best[0]); abl_ho.append(auc(yte, best[2]))
    if len(modnames) < 2:
        continue
    Otr = np.column_stack(Ctr); Ote = np.column_stack(Cte)
    bi = int(np.argmax(abl_oof)); bs_mod, bs_oof, bs_ho = modnames[bi], abl_oof[bi], abl_ho[bi]
    lam = 1.0                                              # ridge-NNLS: optimal non-negative modality weights on OOF
    A = np.vstack([Otr, np.sqrt(lam) * np.eye(Otr.shape[1])]); b = np.concatenate([ytr.astype(float), np.zeros(Otr.shape[1])])
    w, _ = nnls(A, b)
    comb_oof = auc(ytr, Otr @ w / (w.sum() or 1)) if w.sum() > 0 else np.nan
    dom_tot += 1
    if not (comb_oof == comb_oof and comb_oof >= bs_oof - 1e-9):   # FLOOR: never worse than the ablation winner
        w = np.zeros(len(modnames)); w[bi] = 1.0; comb_oof = bs_oof
    else:
        dom_ok += 1
    comb_ho = auc(yte, Ote @ w / (w.sum() or 1))
    top = ", ".join("%s=%.2f" % (modnames[j], w[j] / (w.sum() or 1)) for j in np.argsort(-w)[:3] if w[j] > 0)
    rows.append((mut.replace("mut_", "").replace("cyto_", ""), bs_mod, bs_oof, bs_ho, comb_oof, comb_ho, top))

emit("per mutation: ABLATION best single modality  vs  OPTIMIZED molded weights")
emit("%-14s %-11s %8s %7s   %8s %7s   %s" % ("mutation", "best_mod", "abl_OOF", "abl_HO", "opt_OOF", "opt_HO", "weights"))
for r in rows:
    emit("%-14s %-11s %8.3f %7s   %8.3f %7s   %s" % (r[0][:14], r[1][:11], r[2],
         "%.3f" % r[3] if r[3] == r[3] else "-", r[4], "%.3f" % r[5] if r[5] == r[5] else "-", r[6]))
H = pd.DataFrame(rows, columns=["mut", "bm", "abl_oof", "abl_ho", "opt_oof", "opt_ho", "w"])
emit("\nMEAN  ablation-best: OOF=%.3f HO=%.3f   |   OPTIMIZED molded: OOF=%.3f HO=%.3f"
     % (H.abl_oof.mean(), H.abl_ho.mean(), H.opt_oof.mean(), H.opt_ho.mean()))
emit("dominance on OOF (optimized >= best single modality): %d/%d mutations  [guaranteed]" % (dom_ok, dom_tot))
emit("HO: optimized beats ablation-best on %d/%d mutations" % (int((H.opt_ho > H.abl_ho).sum()), len(H)))
emit("\nMOLDED OK")
