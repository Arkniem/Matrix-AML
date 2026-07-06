#!/usr/bin/env python3
"""FS-method retest for ONE modality (env AMLMM_MODALITY): held-out AUC under
   full features  vs  500-each-way (differential 500/side = 1000)  vs  100-each-way (100/side = 200),
across all 11 models and all mutations. Tells which FS method is best per modality/model.
Full is capped at AMLMM_MAXFEAT (default 8000) so shrLDA/Cell-comm stay feasible.
-> runs/single_modality/_fslevels_<MOD>.txt  +  fslevels_<MOD>.pkl
"""
import os, sys, warnings, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
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

MOD = os.environ.get("AMLMM_MODALITY", "RNA")
MAXFEAT = int(os.environ.get("AMLMM_MAXFEAT", "8000"))
_NJ = int(os.environ.get("AMLMM_NJOBS", "4"))
ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_fslevels_%s.txt" % MOD); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
hold = set(ctx.holdout); samples = ctx.tables["samples"]

def load_block(mod):
    c = ctx.path("_sl_%s.pkl" % mod)
    if os.path.exists(c):
        return pd.read_pickle(c)
    if mod == "Composition":
        return D._sample_level_matrix(ctx, "composition", set(samples.index))
    if mod == "RNA":
        r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0)); mk = pio.udon_markers(ctx, "RNA")
        return r[[g for g in mk if g in r.columns]] if mk else r[list(
            r.loc[[s for s in r.index if s not in hold]].var().sort_values(ascending=False).head(2000).index)]
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

B = load_block(MOD).fillna(0.0); B = B[~B.index.duplicated(keep="first")]
emit("[%s] %d features (full cap=%d)" % (MOD, B.shape[1], MAXFEAT))

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
MODELS = [s[0] for s in SPECS]
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
def diff(Xtr, ytr, kside):
    if Xtr.shape[1] <= 2 * kside:
        return np.arange(Xtr.shape[1])
    F = np.nan_to_num(f_classif(Xtr, ytr)[0]); md = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    o = np.argsort(np.sign(md) * F); return np.unique(np.concatenate([o[:kside], o[-kside:]]))

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
MUTS = []
for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
    y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(B.index); inh = y.index.isin(hold)
    if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
       and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
        MUTS.append(f)

LEVELS = {"full": MAXFEAT, "fs1000": 500, "fs200": 100}      # full->cap; 500/side; 100/side
acc = {mo: {lv: [] for lv in LEVELS} for mo in MODELS}
for mi, m in enumerate(MUTS):
    yall = D._labels_for_field_raw(ctx, m).map(_m01); ym = D.labels_for_field(ctx, m).map(_m01)
    tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2:
        continue
    X0 = B.loc[tr].values; Xt0 = B.loc[te].values; keep = X0.std(0) > 0
    sc = StandardScaler().fit(X0[:, keep]); Xa, Xb = sc.transform(X0[:, keep]), sc.transform(Xt0[:, keep])
    for lv, kside in LEVELS.items():
        sel = diff(Xa, ytr, kside)
        Xtr, Xte = Xa[:, sel], Xb[:, sel]
        for spec in SPECS:
            try: acc[spec[0]][lv].append(roc_auc_score(yte, fit_scores(spec, Xtr, ytr, Xte)))
            except Exception: pass
    emit("  [%2d/%2d] %s" % (mi + 1, len(MUTS), m[:14]))

emit("\n%-9s %8s %8s %8s   %s" % ("model", "full", "500each", "100each", "best"))
for mo in MODELS:
    f = np.mean(acc[mo]["full"]) if acc[mo]["full"] else np.nan
    a = np.mean(acc[mo]["fs1000"]) if acc[mo]["fs1000"] else np.nan
    b = np.mean(acc[mo]["fs200"]) if acc[mo]["fs200"] else np.nan
    best = max([(f, "full"), (a, "500each"), (b, "100each")], key=lambda t: t[0] if t[0] == t[0] else -1)[1]
    emit("%-9s %8.3f %8.3f %8.3f   %s" % (mo, f, a, b, best))
mf = np.nanmean([np.mean(acc[mo]["full"]) for mo in MODELS if acc[mo]["full"]])
ma = np.nanmean([np.mean(acc[mo]["fs1000"]) for mo in MODELS if acc[mo]["fs1000"]])
mb = np.nanmean([np.mean(acc[mo]["fs200"]) for mo in MODELS if acc[mo]["fs200"]])
emit("%-9s %8.3f %8.3f %8.3f" % ("MEAN", mf, ma, mb))
pickle.dump({"modality": MOD, "acc": acc}, open(ctx.path("fslevels_%s.pkl" % MOD), "wb"))
emit("\nFS LEVELS [%s] OK" % MOD)
