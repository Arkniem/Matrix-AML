#!/usr/bin/env python3
"""Lever-attribution ablation: for each model, score held-out AUC under the 2x2 of
{raw features, batch-corrected} x {untuned (fixed), tuned (donor-grouped GridSearchCV)}.
Isolates the effect of BATCH CORRECTION and TUNING per model so we know what to do per model.
Same features + holdout-masked rich labels + sealed 29. -> runs/model_ablation/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

ctx = build_context(Config(run_id="model_ablation"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]
    comp = D._sample_level_matrix(ctx, "composition", set(samples.index))
    markers = pio.udon_markers(ctx, "RNA")
    rna = dataio.sample_modality_matrix(ctx, "RNA")
    rna = np.log1p(rna[[g for g in markers if g in rna.columns]].clip(lower=0))
    feat_raw = comp.join(rna, how="inner").dropna()
    ds = samples["dataset"].astype(str); hold = set(ctx.holdout)
    pool = [s for s in feat_raw.index if s not in hold]
    gmean = feat_raw.loc[pool].mean(axis=0)
    feat_bc = feat_raw.copy()
    for d in ds.loc[feat_raw.index].unique():
        trd = [s for s in pool if ds.get(s) == d]
        if len(trd) >= 4:
            dmean = feat_raw.loc[trd].mean(axis=0)
            rows = [s for s in feat_raw.index if ds.get(s) == d]
            feat_bc.loc[rows] = feat_raw.loc[rows].values - dmean.values + gmean.values
    emit("features %d | raw + batch-corrected built" % feat_raw.shape[1])

    M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(feat_raw.index)
        inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    emit("testable mutations: %d\n" % len(MUTS))

    SPECS = [
        ("logL2",  LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), {"C": [0.02, 0.05, 0.2, 1.0]}, "proba"),
        ("logL1",  LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=3000), {"C": [0.05, 0.1, 0.3]}, "proba"),
        ("elastic", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.1, class_weight="balanced", max_iter=4000), {"C": [0.05, 0.1, 0.3]}, "proba"),
        ("linSVM", LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), {"C": [0.005, 0.02, 0.1]}, "dec"),
        ("shrLDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {"shrinkage": ["auto", 0.3, 0.6]}, "proba"),
        ("PLS",    None, {"n_components": [2, 5, 10]}, "pls"),
        ("RF",     RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=0, n_jobs=1), {"max_depth": [3, 5, None]}, "proba"),
        ("HistGB", HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0), {"max_depth": [2, 3], "max_iter": [150, 250]}, "sw"),
        ("NaiveB", GaussianNB(), {"var_smoothing": [1e-9, 1e-7, 1e-5]}, "proba"),
        ("kNN",    KNeighborsClassifier(n_neighbors=15), {"n_neighbors": [10, 15, 25]}, "proba"),
    ]

    def fit_pred(spec, tuned, Xtr, ytr, grp, Xte):
        name, base, grid, kind = spec
        if kind == "pls":
            c = 5
            if tuned:
                best = -1; k = min(3, len(set(grp)))
                for cc in grid["n_components"]:
                    au = []
                    for tri, vai in GroupKFold(k).split(Xtr, ytr, grp):
                        if len(set(ytr[vai])) < 2:
                            continue
                        mm = PLSRegression(n_components=min(cc, Xtr.shape[1])).fit(Xtr[tri], ytr[tri].astype(float))
                        au.append(roc_auc_score(ytr[vai], mm.predict(Xtr[vai]).ravel()))
                    if au and np.mean(au) > best:
                        best, c = np.mean(au), cc
            m = PLSRegression(n_components=min(c, Xtr.shape[1])).fit(Xtr, ytr.astype(float))
            return m.predict(Xte).ravel()
        if not tuned:
            est = clone(base)
            if kind == "sw":
                est.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            else:
                est.fit(Xtr, ytr)
        else:
            gs = GridSearchCV(clone(base), grid, cv=GroupKFold(min(3, len(set(grp)))), scoring="roc_auc", n_jobs=1)
            if kind == "sw":
                gs.fit(Xtr, ytr, groups=grp, sample_weight=compute_sample_weight("balanced", ytr))
            else:
                gs.fit(Xtr, ytr, groups=grp)
            est = gs.best_estimator_
        return est.decision_function(Xte) if kind == "dec" else est.predict_proba(Xte)[:, 1]

    COND = [("A_raw_fixed", feat_raw, False), ("B_raw_tuned", feat_raw, True),
            ("C_bc_fixed", feat_bc, False), ("D_bc_tuned", feat_bc, True)]
    res = {(n, c[0]): [] for n, _, _, _ in SPECS for c in COND}
    for m in MUTS:
        yall = D._labels_for_field_raw(ctx, m).map(_m01).reindex(feat_raw.index)
        ym = D.labels_for_field(ctx, m).map(_m01).reindex(feat_raw.index)
        train = [s for s in feat_raw.index if pd.notna(ym[s]) and s not in hold]
        test = [s for s in feat_raw.index if s in hold and pd.notna(yall[s])]
        yte = np.array([int(yall[s]) for s in test])
        if len(set(yte)) < 2:
            continue
        ytr = np.array([int(yall[s]) for s in train]); grp = samples.reindex(train)["donor_group"].astype(str).values
        for cname, X, tuned in COND:
            sc = StandardScaler().fit(X.loc[train].values)
            Xtr, Xte = sc.transform(X.loc[train].values), sc.transform(X.loc[test].values)
            for spec in SPECS:
                try:
                    res[(spec[0], cname)].append(roc_auc_score(yte, fit_pred(spec, tuned, Xtr, ytr, grp, Xte)))
                except Exception:
                    res[(spec[0], cname)].append(np.nan)

    emit("%-8s %8s %8s %8s %8s | %8s %8s   %s" % ("model", "raw/fix", "raw/tune", "bc/fix", "bc/tune", "d_tune", "d_batch", "best -> do"))
    for n, _, _, _ in SPECS:
        A = np.nanmean(res[(n, "A_raw_fixed")]); B = np.nanmean(res[(n, "B_raw_tuned")])
        C = np.nanmean(res[(n, "C_bc_fixed")]); Dd = np.nanmean(res[(n, "D_bc_tuned")])
        dtune = B - A; dbatch = C - A
        cells = {"raw,untuned": A, "raw,tuned": B, "batch-corr,untuned": C, "batch-corr,tuned": Dd}
        best = max(cells, key=cells.get)
        emit("%-8s %8.3f %8.3f %8.3f %8.3f | %+8.3f %+8.3f   %.3f (%s)"
             % (n, A, B, C, Dd, dtune, dbatch, cells[best], best))
    emit("\nd_tune = (raw,tuned - raw,fixed): tuning's effect | d_batch = (bc,fixed - raw,fixed): batch-correction's effect")
    emit("\nMODEL ABLATION OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
