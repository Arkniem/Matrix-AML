#!/usr/bin/env python3
"""Early fusion: concatenate ALL modalities into one feature matrix (each modality's top-500
differential features, leakage-safe per mutation), fit the strong models, score sealed held-out.
Compare to the best single modality. Loads caches for big modalities. -> runs/single_modality/_early_fusion.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_early_fusion.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
hold = set(ctx.holdout); samples = ctx.tables["samples"]

def load_block(mod):
    cache = ctx.path("_sl_%s.pkl" % mod)
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    if mod == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(samples.index))
    if mod == "ADT":
        return dataio.sample_modality_matrix(ctx, "ADT")
    if mod == "Lipid":
        return dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
    if mod == "Metabolite":
        return dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
    if mod == "LSC":
        t = ctx.tables.get("lsc_calls"); cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")
    raise SystemExit(mod)

MODS = ["Composition", "RNA", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
BLK = {}
for m in MODS:
    try:
        BLK[m] = load_block(m).fillna(0.0)
        BLK[m] = BLK[m][~BLK[m].index.duplicated(keep="first")]
        emit("loaded %-12s %d feat" % (m, BLK[m].shape[1]))
    except Exception as e:
        emit("skip %s: %s" % (m, e))
common = sorted(set().union(*[set(b.index) for b in BLK.values()]))
BLK = {m: b.reindex(common) for m, b in BLK.items()}

SPECS = [("logL2", LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), "p"),
         ("linSVM", LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), "d"),
         ("shrLDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "p"),
         ("PLS", None, "pls"),
         ("RF", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=0, n_jobs=4), "p")]
def sc_pred(spec, Xtr, ytr, Xte):
    n, base, k = spec
    if k == "pls":
        return PLSRegression(n_components=min(15, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()
    e = clone(base).fit(Xtr, ytr)
    return e.decision_function(Xte) if k == "d" else e.predict_proba(Xte)[:, 1]
def diff(Xtr, ytr, kside):
    if Xtr.shape[1] <= 2 * kside:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0]); md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:kside], o[-kside:]]))

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
MUTS = []
for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
    y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(common); inh = y.index.isin(hold)
    if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
       and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
        MUTS.append(f)

emit("\n%-13s %3s  %s   %7s %7s" % ("mutation", "pos", "  ".join("%-6s" % s[0] for s in SPECS), "FUSEbst", "single*"))
rows = []
for m in MUTS:
    yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
    tr = [s for s in common if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in common if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2:
        continue
    ptr, pte, singles = [], [], []
    for mod, B in BLK.items():
        X0 = B.loc[tr].values; Xt0 = B.loc[te].values; keep = X0.std(0) > 0
        if keep.sum() == 0:
            continue
        sc = StandardScaler().fit(X0[:, keep]); Xa, Xb = sc.transform(X0[:, keep]), sc.transform(Xt0[:, keep])
        sel = diff(Xa, ytr, 250)
        ptr.append(Xa[:, sel]); pte.append(Xb[:, sel])
        try: singles.append(roc_auc_score(yte, sc_pred(("logL2", clone(SPECS[0][1]), "p"), Xa[:, sel], ytr, Xb[:, sel])))
        except Exception: pass
    Xtr = np.hstack(ptr); Xte = np.hstack(pte)
    aucs = []
    for spec in SPECS:
        try: aucs.append(roc_auc_score(yte, sc_pred(spec, Xtr, ytr, Xte)))
        except Exception: aucs.append(np.nan)
    fuse_best = np.nanmax(aucs); single_best = max(singles) if singles else np.nan
    rows.append((m, fuse_best, single_best))
    emit("%-13s %3d  %s   %7.2f %7.2f" % (m.replace("mut_", "").replace("cyto_", "")[:13], int(yte.sum()),
         "  ".join("%6.2f" % a for a in aucs), fuse_best, single_best))

fb = np.nanmean([r[1] for r in rows]); sb = np.nanmean([r[2] for r in rows])
emit("\nMEAN  early-fusion(best model) %.3f   vs  best-single-modality(logL2) %.3f   delta %+.3f" % (fb, sb, fb - sb))
emit("\nEARLY FUSION OK")
