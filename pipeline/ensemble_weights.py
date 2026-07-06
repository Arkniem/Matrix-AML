#!/usr/bin/env python3
"""Phase B — consistency-weighted modality ensemble.
Reads runs/single_modality/preds_<MOD>.pkl (held-out scores + donor-CV reliability per modality/mutation/model).
For each (mutation, model) it weights modalities by CV reliability and combines via weighted rank-average:
  w_margin = max(0, cv_mean - 0.5)                 # deprioritize near-chance modalities
  w_sn     = max(0, cv_mean - 0.5) / (cv_std+0.05) # reward consistency (high mean, low variance)
Compares: CV-best single modality | uniform ensemble | margin-weighted | consistency-weighted.
-> runs/single_modality/_ensemble.txt  +  ensemble_weights.tsv (the learned weights)
"""
import os, sys, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "runs", "single_modality")
RES = os.path.join(RUN, "_ensemble.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")

P = {}
for f in sorted(glob.glob(os.path.join(RUN, "preds_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
emit("modalities loaded: %s" % ", ".join(MODS))
muts = sorted({m for d in P.values() for m in d["data"]})
models = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]

def aligned(mut, model):
    """Return (truth, {mod: ranked_scores}, {mod:(cv_mean,cv_std)}) over the held-out samples
    common to every modality that has this (mut, model)."""
    avail = []
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        mr = rec["models"].get(model)
        if not mr or mr.get("scores") is None:
            continue
        avail.append((mod, dict(zip(rec["test_ids"], mr["scores"])), dict(zip(rec["test_ids"], rec["truth"])),
                      mr.get("cv_mean"), mr.get("cv_std")))
    if len(avail) < 1:
        return None
    common = set(avail[0][1])
    for a in avail[1:]:
        common &= set(a[1])
    common = sorted(common)
    if len(common) < 4:
        return None
    truth = np.array([avail[0][2][s] for s in common])
    if len(set(truth)) < 2:
        return None
    sc, cv = {}, {}
    for mod, smap, tmap, cm, cs in avail:
        v = np.array([smap[s] for s in common], float)
        sc[mod] = rankdata(v)                              # rank-normalize (scale-free across models)
        cv[mod] = (cm if cm is not None else 0.5, cs if cs is not None else 0.25)
    return truth, sc, cv

def comb(truth, sc, weights):
    tot = sum(weights.values())
    if tot <= 0:
        return np.nan
    agg = np.zeros(len(truth))
    for mod, r in sc.items():
        agg += weights.get(mod, 0.0) * r
    return roc_auc_score(truth, agg / tot)

rows = []          # per (mutation, model): the four strategies
wrows = []         # learned weights
for mut in muts:
    for model in models:
        al = aligned(mut, model)
        if al is None:
            continue
        truth, sc, cv = al
        # CV-best single modality -> its held-out AUC
        best_mod = max(sc, key=lambda mo: cv[mo][0])
        best_single = roc_auc_score(truth, sc[best_mod])
        w_uni = {mo: 1.0 for mo in sc}
        w_mar = {mo: max(0.0, cv[mo][0] - 0.5) for mo in sc}
        w_sn = {mo: max(0.0, cv[mo][0] - 0.5) / (cv[mo][1] + 0.05) for mo in sc}
        if sum(w_mar.values()) <= 0:
            w_mar = w_uni
        if sum(w_sn.values()) <= 0:
            w_sn = w_uni
        rows.append((mut, model, best_mod, best_single, comb(truth, sc, w_uni),
                     comb(truth, sc, w_mar), comb(truth, sc, w_sn)))
        for mo in sc:
            wrows.append((mut, model, mo, round(cv[mo][0], 3), round(cv[mo][1], 3),
                          round(w_sn[mo] / (sum(w_sn.values()) or 1), 3)))

df = pd.DataFrame(rows, columns=["mutation", "model", "cv_best_mod", "best_single", "uniform", "w_margin", "w_consistency"])
pd.DataFrame(wrows, columns=["mutation", "model", "modality", "cv_mean", "cv_std", "weight"]).to_csv(
    os.path.join(RUN, "ensemble_weights.tsv"), sep="\t", index=False)

emit("\n%-9s %11s %9s %9s %11s   %s" % ("model", "best_single", "uniform", "w_margin", "w_consist", "consist-best"))
for model in models:
    s = df[df.model == model]
    if not len(s):
        continue
    emit("%-9s %11.3f %9.3f %9.3f %11.3f   %+.3f"
         % (model, s.best_single.mean(), s.uniform.mean(), s.w_margin.mean(), s.w_consistency.mean(),
            s.w_consistency.mean() - s.best_single.mean()))
emit("\n%-9s %11.3f %9.3f %9.3f %11.3f   %+.3f"
     % ("ALL", df.best_single.mean(), df.uniform.mean(), df.w_margin.mean(), df.w_consistency.mean(),
        df.w_consistency.mean() - df.best_single.mean()))

emit("\nper-mutation (best model row): CV-best-single vs consistency-weighted")
emit("%-13s %8s %9s %9s   %s" % ("mutation", "single", "weighted", "delta", "top-weighted modalities"))
for mut in muts:
    s = df[df.mutation == mut]
    if not len(s):
        continue
    r = s.loc[s.w_consistency.idxmax()]
    w = pd.DataFrame(wrows, columns=["mutation", "model", "modality", "cv_mean", "cv_std", "weight"])
    w = w[(w.mutation == mut) & (w.model == r.model)].sort_values("weight", ascending=False)
    top = ", ".join("%s=%.2f" % (x.modality, x.weight) for x in w.head(3).itertuples())
    emit("%-13s %8.2f %9.2f %9.2f   %s" % (mut.replace("mut_", "").replace("cyto_", "")[:13],
         r.best_single, r.w_consistency, r.w_consistency - r.best_single, top))
emit("\nENSEMBLE OK")
