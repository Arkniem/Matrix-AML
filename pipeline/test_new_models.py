#!/usr/bin/env python3
"""Test 20 NEW model types (beyond the existing 11-model roster) on the fused multimodal representation.

Per mutation: z-scored, balanced top-100/modality features concatenated (same representation for all models,
so this ranks MODELS not features). Fit each model, score the sealed held-out, mean AUC over mutations.
Reference rows = the current strong linear models on the SAME fused representation.
-> runs/single_modality/_new_models.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_decomposition import PLSRegression
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, genetics, udon_features as UF

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_new_models.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
samples = ctx.tables["samples"]; hold = set(ctx.holdout)
MODS = ["RNA", "Composition", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]


def load_block(mod):
    if mod == "RNA":
        return UF.canonical_rna(ctx)
    c = ctx.path("_sl_%s.pkl" % mod)
    if os.path.exists(c):
        return pd.read_pickle(c)
    if mod == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(samples.index))
    if mod in ("ADT", "GRN"):
        return dataio.sample_modality_matrix(ctx, mod)
    if mod in ("Lipid", "Metabolite"):
        return dataio.sample_modality_matrix(ctx, mod, min_spearman=0.3)
    if mod == "Cell-comm":
        return dataio.cellcomm_matrix(ctx)
    if mod == "LSC":
        t = ctx.tables.get("lsc_calls"); cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if c in t.columns]
        return t[cols].apply(pd.to_numeric, errors="coerce")


BLK = {}
for m in MODS:
    try:
        b = load_block(m).fillna(0.0); BLK[m] = b[~b.index.duplicated(keep="first")]
    except Exception as e:
        emit("skip %s: %s" % (m, e))
common = sorted(set.intersection(*[set(b.index) for b in BLK.values()]))
emit("fused representation: z-scored balanced top-100/modality | common samples %d\n" % len(common))

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
MUTS = []
for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
    y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(common); inh = pd.Index(common).isin(hold)
    if int((inh & (y == 1)).sum()) >= 3 and int((inh & (y == 0)).sum()) >= 3 \
       and int((~inh & (y == 1)).sum()) >= 5 and int((~inh & (y == 0)).sum()) >= 5:
        MUTS.append(f)


def diff(X, y, k):
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    F = np.nan_to_num(f_classif(X, y)[0]); md = X[y == 1].mean(0) - X[y == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:k // 2], o[-(k // 2):]]))


def auc(y, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if (ok.sum() >= 4 and len(set(y[ok])) == 2) else np.nan


def scores(est, Xtr, ytr, Xte):
    est.fit(Xtr, ytr)
    if hasattr(est, "predict_proba"):
        try:
            return est.predict_proba(Xte)[:, 1]
        except Exception:
            pass
    if hasattr(est, "decision_function"):
        return est.decision_function(Xte)
    return est.predict(Xte).astype(float)


def mk():
    lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)
    rf = RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=0, n_jobs=4)
    hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
    svm = LinearSVC(C=0.02, class_weight="balanced", max_iter=3000)
    return [
        ("ExtraTrees", ExtraTreesClassifier(300, class_weight="balanced_subsample", random_state=0, n_jobs=4)),
        ("GradBoost", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=200, random_state=0)),
        ("Bagging-DT", BaggingClassifier(n_estimators=100, random_state=0, n_jobs=4)),
        ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.5)),
        ("RidgeClf", RidgeClassifier(alpha=10.0, class_weight="balanced")),
        ("SGD-log", SGDClassifier(loss="log_loss", alpha=1e-3, class_weight="balanced", max_iter=3000, random_state=0)),
        ("SGD-mhuber", SGDClassifier(loss="modified_huber", alpha=1e-3, class_weight="balanced", max_iter=3000, random_state=0)),
        ("PassiveAggr", PassiveAggressiveClassifier(C=0.1, class_weight="balanced", max_iter=3000, random_state=0)),
        ("Perceptron", Perceptron(penalty="l2", alpha=1e-3, class_weight="balanced", max_iter=3000, random_state=0)),
        ("GaussProc", GaussianProcessClassifier(1.0 * RBF(1.0), optimizer=None, random_state=0)),
        ("BernoulliNB", BernoulliNB()),
        ("SVC-poly2", SVC(C=1.0, kernel="poly", degree=2, gamma="scale", class_weight="balanced")),
        ("SVC-poly3", SVC(C=1.0, kernel="poly", degree=3, gamma="scale", class_weight="balanced")),
        ("SVC-sigmoid", SVC(C=1.0, kernel="sigmoid", gamma="scale", class_weight="balanced")),
        ("NuSVC-rbf", NuSVC(nu=0.3, kernel="rbf", gamma="scale", class_weight="balanced")),
        ("DecisionTree", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0)),
        ("LDA-svd", LinearDiscriminantAnalysis(solver="svd")),
        ("Bagging-linSVM", BaggingClassifier(svm, n_estimators=50, random_state=0, n_jobs=4)),
        ("Stacking", StackingClassifier([("svm", svm), ("lr", lr), ("rf", rf)],
            final_estimator=LogisticRegression(C=0.1, max_iter=3000), cv=3)),
        ("Voting-soft", VotingClassifier([("lr", lr), ("rf", rf), ("hgb", hgb)], voting="soft")),
        ("[ref]linSVM", LinearSVC(C=0.02, class_weight="balanced", max_iter=5000)),
        ("[ref]logL2", LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)),
    ]


names = [n for n, _ in mk()] + ["[ref]PLS"]
acc = {n: [] for n in names}
for mi, mut in enumerate(MUTS):
    yall = D._labels_for_field_raw(ctx, mut).map(_m01); ym = D.labels_for_field(ctx, mut).map(_m01)
    tr = [s for s in common if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in common if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2 or len(set(ytr)) < 2:
        continue
    Atr, Ate = [], []
    for mod, B in BLK.items():
        Xtr = B.loc[tr].values; Xte = B.loc[te].values; keep = Xtr.std(0) > 0
        if keep.sum() == 0:
            continue
        sc = StandardScaler().fit(Xtr[:, keep]); Ztr = sc.transform(Xtr[:, keep]); Zte = sc.transform(Xte[:, keep])
        s = diff(Ztr, ytr, 100); Atr.append(Ztr[:, s]); Ate.append(Zte[:, s])
    Xtr = np.hstack(Atr); Xte = np.hstack(Ate)
    for name, est in mk():
        try:
            acc[name].append(auc(yte, scores(est, Xtr, ytr, Xte)))
        except Exception:
            pass
    try:
        acc["[ref]PLS"].append(auc(yte, PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)).predict(Xte).ravel()))
    except Exception:
        pass
    emit("  [%2d/%2d] %s" % (mi + 1, len(MUTS), mut[:16]))

emit("\n%-16s %8s  (n)" % ("model", "mean AUC"))
for n in sorted(names, key=lambda x: -(np.mean(acc[x]) if acc[x] else -1)):
    if acc[n]:
        tag = "  <- reference" if n.startswith("[ref]") else ""
        emit("%-16s %8.3f   (%d)%s" % (n, np.mean(acc[n]), len(acc[n]), tag))
emit("\n(fused-representation comparison; deployed late-fusion linSVM+optimized = 0.864)")
emit("\nNEW MODELS OK")
