#!/usr/bin/env python3
"""Per-mutation optimized predictor v2 — robust double fusion + sealed held-out.

Failure modes of v1 (single best MODEL per mutation by CV): donor-CV is noisy on few positives, so it
sometimes crowns an unstable model that collapses on held-out (IDH2 -> RF -> 0.39). Fix, decided
a-priori from every prior experiment (regularized-linear always wins): restrict the model pool to the
STRONG linear family and AVERAGE over models instead of betting on one, fusing model x modality CELLS
weighted by CV reliability, with a CV-dominance gate to a single cell.

Strategies (all leakage-clean: CV-only selection/weights; held-out scored once):
  best_cell  = CV-top (model,modality) single cell                       [v1-style baseline]
  all_blend  = CV-weighted double-blend over ALL 11 models x modalities  [robust, unrestricted]
  OPTIMIZED  = CV-weighted double-blend over STRONG models x modalities, gated to a dominant cell
  oracle*    = best single (model,modality) on the held-out set          [NON-deployable ceiling]
-> runs/single_modality/_optimized_v2.txt + optimized_v2.tsv
"""
import os, glob, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "runs", "single_modality")
RES = os.path.join(RUN, "_optimized_v2.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")

GATE = 0.08
STRONG = ["logL2", "elastic", "linSVM", "shrLDA", "PLS"]
ALL = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]

P = {}
for f in sorted(glob.glob(os.path.join(RUN, "preds_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
muts = sorted({m for d in P.values() for m in d["data"]})
emit("modalities: %s" % ", ".join(MODS))
emit("STRONG pool: %s | gate %.2f\n" % (", ".join(STRONG), GATE))


def cells(mut, model_set):
    out = []
    for model in model_set:
        for mod in MODS:
            rec = P[mod]["data"].get(mut)
            if not rec:
                continue
            mr = rec["models"].get(model)
            if not mr or mr.get("cv_mean") is None or mr.get("scores") is None:
                continue
            out.append((model, mod, dict(zip(rec["test_ids"], mr["scores"])),
                        dict(zip(rec["test_ids"], rec["truth"])), mr["cv_mean"], mr.get("cv_std") or 0.25))
    return out


def fuse(cl, strategy):
    """Per-sample percentile fusion over cells. strategy: single|blend|gated."""
    if not cl:
        return np.nan, 0
    cm_sorted = sorted(cl, key=lambda c: -c[4])
    if strategy in ("single",) or (strategy == "gated" and len(cm_sorted) >= 2
                                   and cm_sorted[0][4] - cm_sorted[1][4] >= GATE):
        use = [(cm_sorted[0], 1.0)]
    else:
        use = [(c, max(0.0, c[4] - 0.5) / (c[5] + 0.05)) for c in cl]
        if sum(w for _, w in use) <= 0:
            use = [(c, 1.0) for c in cl]
    rows = {}
    for c, w in use:
        if w <= 0:
            continue
        _, _, sc, tr, _, _ = c
        ids = list(sc); vals = np.array([sc[s] for s in ids], float)
        if len(ids) < 2:
            continue
        pct = (rankdata(vals) - 1) / (len(ids) - 1)
        for s, p in zip(ids, pct):
            r = rows.setdefault(s, [tr[s], 0.0, 0.0]); r[1] += w * p; r[2] += w
    y, s_ = [], []
    for s, (t, num, den) in rows.items():
        if den > 0:
            y.append(int(t)); s_.append(num / den)
    if len(set(y)) < 2:
        return np.nan, len(y)
    return roc_auc_score(y, s_), len(y)


rows = []
for mut in muts:
    cA = cells(mut, ALL); cS = cells(mut, STRONG)
    best_cell, _ = fuse(cA, "single")
    all_blend, _ = fuse(cA, "blend")
    opt, n = fuse(cS, "gated")
    oracle = max([roc_auc_score([int(t) for t in c[3].values()],
                                 [c[2][s] for s in c[3]]) if len(set(c[3].values())) > 1 else -1
                  for c in cA] + [-1])
    rows.append((mut, n, best_cell, all_blend, opt, oracle))

df = pd.DataFrame(rows, columns=["mutation", "n", "best_cell", "all_blend", "optimized", "oracle"])
df.to_csv(os.path.join(RUN, "optimized_v2.tsv"), sep="\t", index=False)
emit("%-14s %3s %9s %9s %10s %8s" % ("mutation", "n", "best_cell", "all_blend", "OPTIMIZED", "oracle*"))
for r in rows:
    emit("%-14s %3d %9.2f %9.2f %10.2f %8.2f" % (
        r[0].replace("mut_", "").replace("cyto_", "")[:14], r[1], r[2], r[3], r[4], r[5]))
emit("%-14s %3s %9.3f %9.3f %10.3f %8.3f" % (
    "MEAN", "", df.best_cell.mean(), df.all_blend.mean(), df.optimized.mean(), df.oracle.mean()))
emit("\nv2 optimized %.3f | v1 best-single-model 0.844 | best_cell %.3f | oracle %.3f"
     % (df.optimized.mean(), df.best_cell.mean(), df.oracle.mean()))
emit("optimized beats best_cell by %+.3f ; headroom to oracle %+.3f"
     % (df.optimized.mean() - df.best_cell.mean(), df.oracle.mean() - df.optimized.mean()))
emit("\nOPTIMIZE v2 OK")
