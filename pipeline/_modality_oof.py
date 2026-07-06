#!/usr/bin/env python3
"""Phase A-OOF — per-modality OUT-OF-FOLD predictions for leakage-clean modality-weight optimization.

For ONE modality (env AMLMM_MODALITY), all 11 models, all mutations, capture two things per (mutation, model):
  * OOF scores  : donor-grouped GroupKFold over the TRAINING samples only; every train sample gets a
                  prediction from a model trained on folds that do NOT contain it (per-fold, train-only
                  differential feature selection). These honest level-1 features are what the weight
                  optimizer / stacker learns the perfect per-modality weights on.
  * test scores : model refit on the FULL training set, predicting the sealed held-out (== preds_<MOD>).
The held-out samples are NEVER seen during OOF generation. Same compute envelope as Phase A.
-> runs/single_modality/oof_<MOD>.pkl  +  _oof_<MOD>.txt
"""
import os, sys, warnings, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

MOD = os.environ.get("AMLMM_MODALITY", "Composition")
MAXFEAT = int(os.environ.get("AMLMM_MAXFEAT", "8000"))
_NJ = int(os.environ.get("AMLMM_NJOBS", "4"))
ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_oof_%s.txt" % MOD); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
hold = set(ctx.holdout)
samples = ctx.tables["samples"]
dg = samples["donor_group"].astype(str)

def load_block(mod):
    if mod == "RNA":                                       # canonical RNA = raw markers + UDON program fractions (best rep)
        from amlmm import udon_features as _uf
        return _uf.canonical_rna(ctx)
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
    if mod == "GRN":
        return dataio.sample_modality_matrix(ctx, "GRN")
    if mod == "Cell-comm":
        return dataio.cellcomm_matrix(ctx)
    if mod == "LSC":
        t = ctx.tables.get("lsc_calls")
        cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")
    raise SystemExit("unknown modality %s" % mod)

B = load_block(MOD).fillna(0.0)
B = B[~B.index.duplicated(keep="first")]
emit("[%s] %d features, %d samples (cap=%d)" % (MOD, B.shape[1], B.shape[0], MAXFEAT))

SPECS = [
    ("logL2",   LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), "proba"),
    ("logL1",   LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=3000), "proba"),
    ("elastic", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.1, class_weight="balanced", max_iter=4000), "proba"),
    ("linSVM",  LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), "dec"),
    ("shrLDA",  LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "proba"),
    ("PLS",     "pls", "pls"),
    ("RF",      RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=0, n_jobs=_NJ), "proba"),
    ("HistGB",  HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0), "sw"),
    ("NaiveB",  GaussianNB(), "proba"),
    ("kNN",     KNeighborsClassifier(n_neighbors=15), "proba"),
    ("MLP",     MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0), "proba"),
]
def fit_scores(spec, Xtr, ytr, Xte):
    name, base, kind = spec
    if kind == "pls":
        return PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()
    est = clone(base)
    if kind == "sw":
        est.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        est.fit(Xtr, ytr)
    return est.decision_function(Xte) if kind == "dec" else est.predict_proba(Xte)[:, 1]

def diff_cap(Xtr, ytr, k):                                  # differential top-(k/2)/side; train-only per call
    if Xtr.shape[1] <= k:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0])
    md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F)
    return np.unique(np.concatenate([o[:k // 2], o[-(k // 2):]]))

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
MUTS = []
for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
    y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(B.index); inh = y.index.isin(hold)
    if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
       and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
        MUTS.append(f)
emit("testable mutations: %d" % len(MUTS))

OUT = {"modality": MOD, "nfeat": int(B.shape[1]), "data": {}}
for mi, m in enumerate(MUTS):
    yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
    tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2 or len(set(ytr)) < 2:
        continue
    grp = dg.reindex(tr).values
    Braw_tr = B.loc[tr].values; Braw_te = B.loc[te].values
    keep = Braw_tr.std(0) > 0
    Braw_tr, Braw_te = Braw_tr[:, keep], Braw_te[:, keep]
    sc = StandardScaler().fit(Braw_tr)                      # scaler fit on TRAIN only
    Xtr_a, Xte_a = sc.transform(Braw_tr), sc.transform(Braw_te)
    cap = diff_cap(Xtr_a, ytr, MAXFEAT)                     # full-train cap for the held-out refit
    Xtr, Xte = Xtr_a[:, cap], Xte_a[:, cap]

    ng = len(set(grp))
    folds = list(GroupKFold(min(3, ng)).split(Xtr_a, ytr, grp)) if ng >= 2 else []
    fold_data = []                                          # precompute per-fold cap+slices ONCE (not per model)
    for tri, vai in folds:
        if len(set(ytr[tri])) < 2:
            continue
        cp = diff_cap(Xtr_a[tri], ytr[tri], MAXFEAT)        # per-fold cap on the fold's TRAIN rows only
        fold_data.append((tri, vai, Xtr_a[tri][:, cp], Xtr_a[vai][:, cp]))
    rec = {"train_ids": tr, "train_truth": ytr.tolist(),
           "test_ids": te, "test_truth": yte.tolist(), "models": {}}
    for spec in SPECS:
        try:
            test_sco = fit_scores(spec, Xtr, ytr, Xte)      # held-out: refit on full train
        except Exception:
            test_sco = None
        oof = np.full(len(tr), np.nan)
        for tri, vai, Xf_tr, Xf_va in fold_data:
            try:
                oof[vai] = fit_scores(spec, Xf_tr, ytr[tri], Xf_va)
            except Exception:
                pass
        rec["models"][spec[0]] = {
            "oof":  [None if x != x else float(x) for x in oof],
            "test": (None if test_sco is None else [float(x) for x in test_sco]),
        }
    OUT["data"][m] = rec
    emit("  [%2d/%2d] %s done (tr=%d te=%d n+=%d)" % (mi + 1, len(MUTS), m[:14], len(tr), len(te), int(yte.sum())))

with open(ctx.path("oof_%s.pkl" % MOD), "wb") as f:
    pickle.dump(OUT, f)
emit("\nwrote oof_%s.pkl | MODALITY OOF [%s] OK" % (MOD, MOD))
