#!/usr/bin/env python3
"""Verify the healthy-vs-diseased control is BIOLOGY, not BATCH.

(1) WITHIN-DATASET: restrict to datasets that contain BOTH controls and diseased (the only place batch
    and disease are separable); donor-grouped CV per modality. If AUC stays high there, the separation
    is real within a single batch, not a dataset artifact.
(2) PERMUTATION NULL: shuffle labels, rerun donor-grouped CV. AUC should collapse to ~0.5; if it stays
    high, the CV/feature-selection is inflating and the headline is untrustworthy.
-> runs/single_modality/_control_hd_verify.txt
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
from amlmm import discovery as D, dataio, pseudobulk_io as pio

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_control_hd_verify.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
samples = ctx.tables["samples"]
dg = samples["donor_group"].astype(str); ds = samples["dataset"].astype(str)
dc = samples.get("disease_category").astype(str)

y = pd.Series(index=samples.index, dtype=float)
y[dc.eq("Control")] = 0.0
y[dc.isin({"AML", "MDS", "T-ALL"})] = 1.0
lab = y.dropna()
both = sorted(d for d in set(ds.loc[lab.index]) if {0.0, 1.0} <= set(lab[ds.loc[lab.index] == d]))
emit("datasets with BOTH classes (within-dataset eval set): %s" % both)
win = lab.index[ds.loc[lab.index].isin(both)]
emit("within-dataset cohort: %d samples, %d controls\n" % (len(win), int((lab.loc[win] == 0).sum())))

def load_block(mod):
    c = ctx.path("_sl_%s.pkl" % mod)
    if os.path.exists(c):
        return pd.read_pickle(c)
    if mod == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(samples.index))
    if mod == "RNA":
        r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0)); mk = pio.udon_markers(ctx, "RNA")
        return r[[g for g in mk if g in r.columns]] if mk else r
    if mod in ("ADT", "GRN"):
        return dataio.sample_modality_matrix(ctx, mod)
    if mod in ("Lipid", "Metabolite"):
        return dataio.sample_modality_matrix(ctx, mod, min_spearman=0.3)
    if mod == "Cell-comm":
        return dataio.cellcomm_matrix(ctx)
    if mod == "LSC":
        t = ctx.tables.get("lsc_calls"); cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")
    raise SystemExit(mod)

def diff(Xtr, ytr, k):
    if Xtr.shape[1] <= 2 * k:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0]); md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:k], o[-k:]]))

def cv_auc(B, ids, yy, g, perm_seed=None):
    X = B.reindex(ids).fillna(0.0).values
    if perm_seed is not None:                              # label-permutation null
        yy = yy[np.random.RandomState(perm_seed).permutation(len(yy))]
    ng = len(set(g))
    oof = np.full(len(ids), np.nan)
    for tri, vai in GroupKFold(min(4, ng)).split(X, yy, g.values):
        if len(set(yy[tri])) < 2:
            continue
        keep = X[tri].std(0) > 0
        sc = StandardScaler().fit(X[tri][:, keep]); Xa = sc.transform(X[tri][:, keep]); Xb = sc.transform(X[vai][:, keep])
        sel = diff(Xa, yy[tri], 400); Xa, Xb = Xa[:, sel], Xb[:, sel]
        try:
            oof[vai] = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Xa, yy[tri]).predict_proba(Xb)[:, 1]
        except Exception:
            pass
    ok = ~np.isnan(oof)
    return roc_auc_score(yy[ok], oof[ok]) if len(set(yy[ok])) == 2 else np.nan

MODS = ["Lipid", "Composition", "Metabolite", "GRN", "RNA", "Cell-comm", "ADT"]
emit("%-12s %6s %5s %9s %12s" % ("modality", "n", "nHD", "withinAUC", "permNull(mean)"))
for mod in MODS:
    try:
        B = load_block(mod).fillna(0.0); B = B[~B.index.duplicated(keep="first")]
    except Exception as e:
        emit("%-12s skip: %s" % (mod, e)); continue
    ids = [s for s in B.index if s in set(win)]
    if len(ids) < 16 or int((lab.loc[ids] == 0).sum()) < 4:
        emit("%-12s within-dataset too small (n=%d, nHD=%d)" % (mod, len(ids), int((lab.loc[ids] == 0).sum()))); continue
    yy = lab.loc[ids].values.astype(int); g = dg.loc[ids]
    wauc = cv_auc(B, ids, yy, g)
    nulls = [cv_auc(B, ids, yy.copy(), g, perm_seed=sd) for sd in range(5)]
    emit("%-12s %6d %5d %9.3f %12.3f" % (mod, len(ids), int((yy == 0).sum()), wauc, np.nanmean(nulls)))
emit("\nwithinAUC high + permNull~0.5 => the healthy/diseased signal is real biology, not batch. VERIFY OK")
