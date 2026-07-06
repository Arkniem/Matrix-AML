#!/usr/bin/env python3
"""Model-structure panel: per mutation, train a battery of classifiers on the SAME features
(composition + RNA marker genes), holdout-masked (rich explicit labels), and score held-out AUC on the
sealed 29. Extends the NN/logistic comparison to: L2/L1/elastic-net logistic, linear SVM, random forest,
hist gradient boosting, PLS-DA, shrinkage LDA, naive Bayes, kNN. CPU job. -> runs/model_panel/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio

ctx = build_context(Config(run_id="model_panel"))
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
    emit("features: %d (composition %d + RNA-markers %d) | samples %d"
         % (feat.shape[1], comp.shape[1], rna.shape[1], feat.shape[0]))

    from amlmm import genetics
    M = ctx.tables.get("mutations")
    if M is None:
        M = genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    held_set = set(ctx.holdout)
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(feat.index)
        inh = y.index.isin(held_set)
        hp, hn = int(((inh) & (y == 1)).sum()), int(((inh) & (y == 0)).sum())
        tp, tn = int(((~inh) & (y == 1)).sum()), int(((~inh) & (y == 0)).sum())
        if hp >= 3 and hn >= 3 and tp >= 5 and tn >= 5:
            MUTS.append(f)
    emit("testable withheld mutation flags (>=3 held-out pos & neg, >=5 train pos & neg): %d" % len(MUTS))
    emit("  " + ", ".join(MUTS))

    def mk_models():
        return [
            ("logL2",    LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), "proba"),
            ("logL1",    LogisticRegression(penalty="l1", C=0.1, solver="liblinear", class_weight="balanced", max_iter=3000), "proba"),
            ("elastic",  LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", class_weight="balanced", max_iter=4000), "proba"),
            ("linSVM",   LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), "dec"),
            ("RF",       RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=0, n_jobs=1), "proba"),
            ("HistGB",   HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0), "proba_sw"),
            ("PLS-DA",   PLSRegression(n_components=5), "pls"),
            ("shrLDA",   LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "proba"),
            ("NaiveB",   GaussianNB(), "proba"),
            ("kNN",      KNeighborsClassifier(n_neighbors=15), "proba"),
        ]
    NAMES = [n for n, _, _ in mk_models()]

    def fit_score(est, kind, Xtr, ytr, Xte):
        if kind == "pls":
            est.fit(Xtr, ytr.astype(float)); return est.predict(Xte).ravel()
        if kind == "proba_sw":
            est.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr)); return est.predict_proba(Xte)[:, 1]
        est.fit(Xtr, ytr)
        return est.decision_function(Xte) if kind == "dec" else est.predict_proba(Xte)[:, 1]

    def to01(s):
        return s.map({"present": 1.0, "absent": 0.0})

    results = {n: [] for n in NAMES}
    emit("\n%-12s %4s %4s  %s" % ("mutation", "n", "pos", "  ".join("%-7s" % n for n in NAMES)))
    for m in MUTS:
        yall = to01(D._labels_for_field_raw(ctx, m)).reindex(feat.index)
        ymask = to01(D.labels_for_field(ctx, m)).reindex(feat.index)
        train = [s for s in feat.index if pd.notna(ymask[s]) and s not in ctx.holdout]
        test = [s for s in feat.index if s in ctx.holdout and pd.notna(yall[s])]
        yte = np.array([int(yall[s]) for s in test])
        if len(test) < 4 or len(set(yte)) < 2 or pd.Series([yall[s] for s in train]).nunique() < 2:
            emit("%-12s  (insufficient)" % m); continue
        sc = StandardScaler().fit(feat.loc[train].values)
        Xtr = sc.transform(feat.loc[train].values); Xte = sc.transform(feat.loc[test].values)
        ytr = np.array([int(yall[s]) for s in train])
        row = []
        for n, est, kind in mk_models():
            try:
                a = roc_auc_score(yte, fit_score(est, kind, Xtr, ytr, Xte)); results[n].append(a); row.append("%7.2f" % a)
            except Exception as e:
                row.append("%7s" % "err")
        emit("%-12s %4d %4d  %s" % (m, len(test), int(yte.sum()), "  ".join(row)))

    emit("\n%-12s      %s" % ("MEAN AUC", "  ".join("%7.3f" % (np.mean(results[n]) if results[n] else float("nan")) for n in NAMES)))
    rank = sorted([(np.mean(v), n) for n, v in results.items() if v], reverse=True)
    emit("\nranked:  " + " > ".join("%s %.3f" % (n, a) for a, n in rank))
    emit("(reference from earlier runs: multi-task NN 0.56, separate NN 0.64)")
    emit("\nMODEL PANEL OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
