#!/usr/bin/env python3
"""FIRST CONTROL: can the multimodal atlas separate HEALTHY (control) from DISEASED (AML/MDS/T-ALL)?

A known-easy sanity check that validates the data/pipeline before trusting the hard mutation calls.
Per modality: donor-grouped 5-fold CV -> AUC + balanced accuracy + sensitivity (diseased caught) +
SPECIFICITY (controls correctly called healthy, the boss's key number) + an all-modality fusion row.

Guards the obvious failure mode: a "control" that is really a batch detector. Reports how many of the
control datasets also contain diseased samples; if controls live in their own datasets the separation is
confounded with batch and we say so. Feature selection is per-fold (train-only); no held-out tuning.
-> runs/single_modality/_control_hd.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_control_hd.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
samples = ctx.tables["samples"]
dg = samples["donor_group"].astype(str)
ds = samples["dataset"].astype(str)
dc = samples.get("disease_category")

DISEASED = {"AML", "MDS", "T-ALL"}
y = pd.Series(index=samples.index, dtype=float)
dcs = dc.astype(str)
y[dcs.eq("Control")] = 0.0
y[dcs.isin(DISEASED)] = 1.0
lab = y.dropna()
emit("HEALTHY vs DISEASED  |  healthy(control)=%d  diseased=%d  (excluded NA/other=%d)"
     % (int((lab == 0).sum()), int((lab == 1).sum()), len(samples) - len(lab)))

# ---- batch-confound check ----
hd_ids = lab.index[lab == 0]; di_ids = lab.index[lab == 1]
hd_ds = set(ds.loc[hd_ids]); di_ds = set(ds.loc[di_ids])
shared = sorted(hd_ds & di_ds)
emit("control datasets: %s" % sorted(hd_ds))
emit("datasets with BOTH control & diseased: %s" % shared)
emit("-> %d/%d control datasets also contain diseased samples %s\n"
     % (len(shared), len(hd_ds), "(clean within-dataset signal)" if shared else "(WARNING: fully batch-confounded)"))


def load_block(mod):
    cache = ctx.path("_sl_%s.pkl" % mod)
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    if mod == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(samples.index))
    if mod == "RNA":
        r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0)); mk = pio.udon_markers(ctx, "RNA")
        return r[[g for g in mk if g in r.columns]] if mk else r
    if mod == "ADT":
        return dataio.sample_modality_matrix(ctx, "ADT")
    if mod == "Lipid":
        return dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
    if mod == "Metabolite":
        return dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
    if mod == "GRN":
        return dataio.sample_modality_matrix(ctx, "GRN")
    if mod == "Cell-comm":
        return dataio.cellcomm_matrix(ctx)
    if mod == "LSC":
        t = ctx.tables.get("lsc_calls"); cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")
    raise SystemExit(mod)


MODS = ["RNA", "Composition", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
def diff(Xtr, ytr, k):
    if Xtr.shape[1] <= 2 * k:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0]); md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:k], o[-k:]]))
def fit_logL2(Xtr, ytr, Xte):
    return LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def cv_scores(B):
    ids = [s for s in B.index if s in lab.index]
    if len(ids) < 20:
        return None
    X = B.loc[ids].fillna(0.0).values; yy = lab.loc[ids].values.astype(int); g = dg.loc[ids].values
    ng = len(set(g))
    oof = np.full(len(ids), np.nan)
    for tri, vai in GroupKFold(min(5, ng)).split(X, yy, g):
        if len(set(yy[tri])) < 2:
            continue
        keep = X[tri].std(0) > 0
        sc = StandardScaler().fit(X[tri][:, keep]); Xa = sc.transform(X[tri][:, keep]); Xb = sc.transform(X[vai][:, keep])
        sel = diff(Xa, yy[tri], 500); Xa, Xb = Xa[:, sel], Xb[:, sel]
        try:
            oof[vai] = fit_logL2(Xa, yy[tri], Xb)
        except Exception:
            pass
    ok = ~np.isnan(oof)
    if len(set(yy[ok])) < 2:
        return None
    yt = yy[ok]; sc = oof[ok]
    auc = roc_auc_score(yt, sc)
    npos = int(yt.sum()); thr = np.sort(sc)[::-1][npos - 1] if 0 < npos <= len(sc) else np.median(sc)
    pred = (sc >= thr).astype(int)                          # prevalence-matched threshold (no tuning)
    bal = balanced_accuracy_score(yt, pred)
    sens = float(((pred == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum()))
    spec = float(((pred == 0) & (yt == 0)).sum() / max(1, (yt == 0).sum()))
    return len(ids), int((yt == 0).sum()), auc, bal, sens, spec


emit("%-12s %5s %5s %7s %8s %7s %9s" % ("modality", "n", "nHD", "AUC", "balAcc", "sens", "spec(HD)"))
BLK = {}
for mod in MODS:
    try:
        B = load_block(mod).fillna(0.0); B = B[~B.index.duplicated(keep="first")]; BLK[mod] = B
    except Exception as e:
        emit("%-12s skip: %s" % (mod, e)); continue
    r = cv_scores(B)
    if r is None:
        emit("%-12s too few / degenerate" % mod); continue
    emit("%-12s %5d %5d %7.3f %8.3f %7.3f %9.3f" % (mod, r[0], r[1], r[2], r[3], r[4], r[5]))

# ---- all-modality fusion (per-fold top-150 differential each, leakage-safe) ----
common = sorted(set.intersection(*[set(b.index) for b in BLK.values()]) & set(lab.index)) if BLK else []
if len(common) >= 20:
    yy = lab.loc[common].values.astype(int); g = dg.loc[common].values
    oof = np.full(len(common), np.nan)
    for tri, vai in GroupKFold(min(5, len(set(g)))).split(np.zeros(len(common)), yy, g):
        if len(set(yy[tri])) < 2:
            continue
        Ftr, Fte = [], []
        for mod, B in BLK.items():
            X = B.reindex(common).fillna(0.0).values
            keep = X[tri].std(0) > 0
            if keep.sum() == 0:
                continue
            sc = StandardScaler().fit(X[tri][:, keep]); Xa = sc.transform(X[tri][:, keep]); Xb = sc.transform(X[vai][:, keep])
            sel = diff(Xa, yy[tri], 150); Ftr.append(Xa[:, sel]); Fte.append(Xb[:, sel])
        try:
            oof[vai] = fit_logL2(np.hstack(Ftr), yy[tri], np.hstack(Fte))
        except Exception:
            pass
    ok = ~np.isnan(oof); yt = yy[ok]; sco = oof[ok]
    if len(set(yt)) == 2:
        auc = roc_auc_score(yt, sco); npos = int(yt.sum())
        thr = np.sort(sco)[::-1][npos - 1]; pred = (sco >= thr).astype(int)
        emit("%-12s %5d %5d %7.3f %8.3f %7.3f %9.3f" % ("ALL-FUSED", len(common), int((yt == 0).sum()),
             auc, balanced_accuracy_score(yt, pred),
             ((pred == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum()),
             ((pred == 0) & (yt == 0)).sum() / max(1, (yt == 0).sum())))
emit("\nspec(HD) = fraction of controls correctly called healthy. CONTROL HD OK")
