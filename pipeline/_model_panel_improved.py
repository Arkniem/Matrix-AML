#!/usr/bin/env python3
"""Improved model panel: applies the levers on top of the baseline structure comparison.
  (1) BATCH CORRECTION  - additive per-dataset centering, fit on TRAIN only (removes "which lab"),
  (2) HYPERPARAMETER TUNING - donor-grouped GridSearchCV on the training pool per mutation,
  (3) ENSEMBLE - rank-average of the top linear models,
  (4) BOOTSTRAP 95% CIs on each model's mean AUC.
Same features (composition + RNA markers), holdout-masked rich labels, sealed 29.
(Per-cell-state features = Deploy, run separately.) CPU job. -> runs/model_panel_improved/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import rankdata
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

ctx = build_context(Config(run_id="model_panel_improved"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]; sk_all = list(samples.index)
    comp = D._sample_level_matrix(ctx, "composition", set(sk_all))
    markers = pio.udon_markers(ctx, "RNA")
    rna = dataio.sample_modality_matrix(ctx, "RNA")
    rna = np.log1p(rna[[g for g in markers if g in rna.columns]].clip(lower=0))
    feat = comp.join(rna, how="inner").dropna()
    ds = samples["dataset"].astype(str)
    hold = set(ctx.holdout)

    # ---- (1) batch correction: additive per-dataset centering, fit on TRAIN pool only ----
    pool = [s for s in feat.index if s not in hold]
    gmean = feat.loc[pool].mean(axis=0)
    Xb = feat.copy()
    nfix = 0
    for d in ds.loc[feat.index].unique():
        trd = [s for s in pool if ds.get(s) == d]
        if len(trd) >= 4:
            dmean = feat.loc[trd].mean(axis=0)
            rows = [s for s in feat.index if ds.get(s) == d]
            Xb.loc[rows] = feat.loc[rows].values - dmean.values + gmean.values
            nfix += 1
    feat = Xb
    emit("features %d | samples %d | batch-corrected datasets %d" % (feat.shape[1], feat.shape[0], nfix))

    M = ctx.tables.get("mutations")
    if M is None:
        M = genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(feat.index)
        inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    emit("testable withheld mutations: %d" % len(MUTS))

    def grids():
        return [
            ("logL2",  LogisticRegression(class_weight="balanced", max_iter=3000), {"C": [0.02, 0.05, 0.2]}, "proba"),
            ("linSVM", LinearSVC(class_weight="balanced", max_iter=5000), {"C": [0.005, 0.02, 0.1]}, "dec"),
            ("shrLDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}, "proba"),
            ("RF",     RandomForestClassifier(class_weight="balanced_subsample", random_state=0, n_jobs=1),
                       {"n_estimators": [300], "max_depth": [4, None]}, "proba"),
            ("HistGB", HistGradientBoostingClassifier(random_state=0, l2_regularization=1.0),
                       {"max_depth": [3], "max_iter": [150]}, "proba_sw"),
        ]
    ENS = ["logL2", "linSVM", "shrLDA", "PLS"]            # the ensemble members (top linear) by rank-average
    NAMES = [n for n, _, _, _ in grids()] + ["PLS", "ENSEMBLE"]

    def tune_score(base, grid, kind, Xtr, ytr, grp, Xte):
        if grid:
            k = min(3, len(set(grp)))
            from sklearn.utils.class_weight import compute_sample_weight
            if kind == "proba_sw":
                gs = GridSearchCV(base, grid, cv=GroupKFold(k), scoring="roc_auc", n_jobs=1)
                gs.fit(Xtr, ytr, groups=grp, sample_weight=compute_sample_weight("balanced", ytr))
                return gs.best_estimator_.predict_proba(Xte)[:, 1]
            gs = GridSearchCV(base, grid, cv=GroupKFold(k), scoring="roc_auc", n_jobs=1)
            gs.fit(Xtr, ytr, groups=grp)
            est = gs.best_estimator_
        else:
            est = base.fit(Xtr, ytr)
        return est.decision_function(Xte) if kind == "dec" else est.predict_proba(Xte)[:, 1]

    def pls_score(Xtr, ytr, grp, Xte):
        best_c, best = 2, -1
        k = min(3, len(set(grp)))
        for c in [2, 5, 10]:
            au = []
            for tri, vai in GroupKFold(k).split(Xtr, ytr, grp):
                if len(set(ytr[vai])) < 2:
                    continue
                m = PLSRegression(n_components=min(c, Xtr.shape[1])).fit(Xtr[tri], ytr[tri].astype(float))
                au.append(roc_auc_score(ytr[vai], m.predict(Xtr[vai]).ravel()))
            if au and np.mean(au) > best:
                best, best_c = np.mean(au), c
        m = PLSRegression(n_components=min(best_c, Xtr.shape[1])).fit(Xtr, ytr.astype(float))
        return m.predict(Xte).ravel()

    per = {n: [] for n in NAMES}
    emit("\n%-12s %4s  %s" % ("mutation", "pos", "  ".join("%-8s" % n for n in NAMES)))
    for m in MUTS:
        yall = D._labels_for_field_raw(ctx, m).map(_m01).reindex(feat.index)
        ym = D.labels_for_field(ctx, m).map(_m01).reindex(feat.index)
        train = [s for s in feat.index if pd.notna(ym[s]) and s not in hold]
        test = [s for s in feat.index if s in hold and pd.notna(yall[s])]
        yte = np.array([int(yall[s]) for s in test])
        if len(set(yte)) < 2:
            continue
        sc = StandardScaler().fit(feat.loc[train].values)
        Xtr, Xte = sc.transform(feat.loc[train].values), sc.transform(feat.loc[test].values)
        ytr = np.array([int(yall[s]) for s in train]); grp = samples.reindex(train)["donor_group"].astype(str).values
        scores = {}
        for n, base, grid, kind in grids():
            try:
                s = tune_score(base, grid, kind, Xtr, ytr, grp, Xte); scores[n] = s
                per[n].append(roc_auc_score(yte, s))
            except Exception:
                per[n].append(np.nan)
        try:
            sp = pls_score(Xtr, ytr, grp, Xte); scores["PLS"] = sp; per["PLS"].append(roc_auc_score(yte, sp))
        except Exception:
            per["PLS"].append(np.nan)
        ranks = [rankdata(scores[n]) for n in ENS if n in scores]
        ens = np.mean(ranks, axis=0) if ranks else np.zeros(len(yte))
        per["ENSEMBLE"].append(roc_auc_score(yte, ens))
        emit("%-12s %4d  %s" % (m[:12], int(yte.sum()),
             "  ".join("%8.2f" % per[n][-1] for n in NAMES)))

    emit("\n%-12s %4s  %s" % ("MEAN", "", "  ".join("%8.3f" % np.nanmean(per[n]) for n in NAMES)))
    rng = np.random.RandomState(0)
    emit("\nmean AUC with bootstrap 95%% CI (resampling mutations):")
    for n in NAMES:
        v = np.array([x for x in per[n] if x == x])
        if len(v) == 0:
            continue
        boot = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(2000)]
        emit("  %-9s %.3f  [%.3f, %.3f]" % (n, v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
    emit("\nreference (baseline panel, untuned, no batch correction): linSVM 0.840, shrLDA 0.827, logL2 0.820")
    emit("\nIMPROVED PANEL OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
