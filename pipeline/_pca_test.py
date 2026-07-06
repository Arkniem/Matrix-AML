#!/usr/bin/env python3
"""Does a latent (PCA) representation beat raw features per modality? raw vs PCA-30 vs PCA-100,
strong linear models, per mutation, sealed held-out. -> runs/single_modality/_pca_test.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, genetics

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_pca_test.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
hold = set(ctx.holdout); samples = ctx.tables["samples"]

def load_block(mod):
    c = ctx.path("_sl_%s.pkl" % mod)
    if os.path.exists(c):
        return pd.read_pickle(c)
    if mod == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(samples.index))
    if mod == "ADT":
        return dataio.sample_modality_matrix(ctx, "ADT")
    if mod == "Lipid":
        return dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
    if mod == "Metabolite":
        return dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
    if mod == "LSC":
        t = ctx.tables.get("lsc_calls"); cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")
    raise SystemExit(mod)

MODS = ["RNA", "GRN", "Metabolite", "Lipid", "ADT", "Cell-comm", "Composition", "LSC"]
def diff(Xtr, ytr, kside):
    if Xtr.shape[1] <= 2 * kside:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0]); md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:kside], o[-kside:]]))
def lin_auc(Xtr, ytr, Xte, yte):
    a = []
    for mk in ("L", "P"):
        try:
            if mk == "L":
                e = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Xtr, ytr); s = e.predict_proba(Xte)[:, 1]
            else:
                s = PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()
            a.append(roc_auc_score(yte, s))
        except Exception:
            pass
    return max(a) if a else np.nan

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
emit("%-12s %8s %8s %8s   %s" % ("modality", "raw", "PCA-30", "PCA-100", "best"))
for mod in MODS:
    try:
        B = load_block(mod).fillna(0.0); B = B[~B.index.duplicated(keep="first")]
    except Exception as e:
        emit("%-12s skip: %s" % (mod, e)); continue
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(B.index); inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    raw, p30, p100 = [], [], []
    for m in MUTS:
        yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
        tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
        te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
        ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
        if len(set(yte)) < 2:
            continue
        X0 = B.loc[tr].values; Xt0 = B.loc[te].values; keep = X0.std(0) > 0
        sc = StandardScaler().fit(X0[:, keep]); Xa, Xb = sc.transform(X0[:, keep]), sc.transform(Xt0[:, keep])
        sel = diff(Xa, ytr, 2000); Xa, Xb = Xa[:, sel], Xb[:, sel]   # cap raw at 4000 for tractability
        raw.append(lin_auc(Xa, ytr, Xb, yte))
        for nc, store in ((30, p30), (100, p100)):
            k = min(nc, Xa.shape[1] - 1, len(tr) - 1)
            if k < 2:
                store.append(np.nan); continue
            pca = PCA(n_components=k, random_state=0).fit(Xa)
            store.append(lin_auc(pca.transform(Xa), ytr, pca.transform(Xb), yte))
    r, a30, a100 = np.nanmean(raw), np.nanmean(p30), np.nanmean(p100)
    best = max(r, a30, a100); tag = {r: "raw", a30: "PCA-30", a100: "PCA-100"}[best]
    emit("%-12s %8.3f %8.3f %8.3f   %s" % (mod, r, a30, a100, tag))
emit("\nPCA TEST OK")
