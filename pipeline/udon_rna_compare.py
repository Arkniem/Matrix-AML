#!/usr/bin/env python3
"""Does the UDON RNA representation beat raw-expression RNA? Fair comparison (SAME samples + folds):
   RNA-raw (log1p expr on UDON markers) | UDON-fold (log1p control-normalized fold) | UDON-prog (16 programs)
   | raw+prog (raw expression + program fractions).
Tasks: healthy-vs-diseased + FLT3 / NPM1 / TP53 / IDH2 / DNMT3A. Donor-grouped 5-fold CV AUC.
-> runs/single_modality/_udon_compare.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, udon_features as UF

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_udon_compare.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
samples = ctx.tables["samples"]; dg = samples["donor_group"].astype(str)

def rna_raw():
    r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0)); mk = pio.udon_markers(ctx, "RNA")
    return (r[[g for g in mk if g in r.columns]] if mk else r)

REPS = {}
REPS["RNA-raw"] = rna_raw().fillna(0.0)
REPS["UDON-fold"] = UF.udon_fold_sample_matrix(ctx, markers=True, log=True).fillna(0.0)
REPS["UDON-prog"] = UF.udon_program_matrix(ctx).fillna(0.0)
REPS["raw+prog"] = REPS["RNA-raw"].join(REPS["UDON-prog"], how="left").fillna(0.0)
for k, v in REPS.items():
    REPS[k] = v[~v.index.duplicated(keep="first")]
    emit("rep %-10s %5d feat  %4d samples" % (k, REPS[k].shape[1], REPS[k].shape[0]))
common = set.intersection(*[set(v.index) for v in REPS.values()])
emit("common samples across all reps: %d\n" % len(common))

_m01 = {"present": 1.0, "absent": 0.0}
def hd():
    dc = samples.get("disease_category").astype(str); y = pd.Series(index=samples.index, dtype=float)
    y[dc.eq("Control")] = 0.0; y[dc.isin({"AML", "MDS", "T-ALL"})] = 1.0; return y.dropna()
TASKS = {"healthy-vs-dis": hd()}
for m in ["mut_FLT3", "mut_NPM1", "mut_TP53", "mut_IDH2", "mut_DNMT3A"]:
    TASKS[m.replace("mut_", "")] = D._labels_for_field_raw(ctx, m).map(_m01).dropna()

def diff(Xtr, ytr, k):
    if Xtr.shape[1] <= 2 * k:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0]); md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:k], o[-k:]]))

def auc_on(B, ids, y, folds):
    X = B.reindex(ids).fillna(0.0).values; oof = np.full(len(ids), np.nan)
    for tri, vai in folds:
        if len(set(y[tri])) < 2:
            continue
        keep = X[tri].std(0) > 0
        if keep.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tri][:, keep]); Xa = sc.transform(X[tri][:, keep]); Xb = sc.transform(X[vai][:, keep])
        sel = diff(Xa, y[tri], 300); Xa, Xb = Xa[:, sel], Xb[:, sel]
        try:
            oof[vai] = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Xa, y[tri]).predict_proba(Xb)[:, 1]
        except Exception:
            pass
    ok = ~np.isnan(oof)
    return roc_auc_score(y[ok], oof[ok]) if len(set(y[ok])) == 2 else np.nan

emit("%-16s %4s  %s" % ("task", "n", "  ".join("%-9s" % r for r in REPS)))
for t, lab in TASKS.items():
    ids = [s for s in sorted(common) if s in lab.index]
    y = np.array([int(lab[s]) for s in ids]) if ids else np.array([])
    if len(ids) < 24 or len(set(y)) < 2 or min((y == 0).sum(), (y == 1).sum()) < 5:
        emit("%-16s %4d  (too few)" % (t[:16], len(ids))); continue
    g = dg.loc[ids].values
    folds = [(tri, vai) for tri, vai in GroupKFold(min(5, len(set(g)))).split(np.zeros(len(ids)), y, g)
             if len(set(y[tri])) == 2 and len(set(y[vai])) == 2]
    cells = []
    for r, B in REPS.items():
        a = auc_on(B, ids, y, folds); cells.append("%.3f" % a if a == a else "  -  ")
    emit("%-16s %4d  %s" % (t[:16], len(ids), "  ".join("%-9s" % c for c in cells)))
emit("\nUDON COMPARE OK")
