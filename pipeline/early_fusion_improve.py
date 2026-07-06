#!/usr/bin/env python3
"""Try EVERY tractable early-fusion improvement, fair head-to-head on the sealed held-out (new raw+prog RNA).

Per mutation: per-modality top-k differential features (train-only), then each variant builds its design and
fits; held-out AUC. All variants share the same train/held-out split and feature pool, so they're comparable.

Variants:
  naive      concat raw top-500/modality, logL2                     (current early fusion baseline)
  zscore     z-score + balanced top-200/modality, concat, logL2     (A1+A3 scaling + balanced caps)
  blocknorm  zscore then each block scaled to unit norm, logL2      (A2 block-norm equalization)
  pca        per-modality PCA-20 on z-blocks, concat, logL2         (A4 intermediate fusion)
  residual   residualize imputed blocks vs RNA (Ridge), logL2       (B6 de-redundancy)
  drop       only RNA+Composition+Cell-comm+LSC, logL2              (B8 drop RNA-derived)
  mkl        scale each block by its train-CV reliability, logL2    (D12 weighted early fusion)
  interact   top-3/modality + all pairwise cross-modality products  (E14 explicit interactions)
  histgb     concat z-blocks, HistGB                                (E16 tree interactions)
  rbfsvm     concat z-blocks, RBF-SVM                               (E17 kernel interactions)
  mlp        concat z-blocks, MLP                                   (E18 neural)
  gate       only modalities with train-CV AUC>0.55, logL2          (F19 per-mutation gating)
  late       per-modality linSVM, reliability-weighted late fusion  (late-fusion reference ~0.86)
  hybrid     0.5*zscore-early + 0.5*late                            (G21 early+late hybrid)
-> runs/single_modality/_early_improve.txt
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics, udon_features as UF

ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_early_improve.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")
samples = ctx.tables["samples"]; dg = samples["donor_group"].astype(str); hold = set(ctx.holdout)
MODS = ["RNA", "Composition", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
INDEP = {"RNA", "Composition", "Cell-comm", "LSC"}          # not imputed-from-RNA
IMPUTED = {"ADT", "Lipid", "Metabolite", "GRN"}


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
emit("loaded: " + ", ".join("%s(%d)" % (m, BLK[m].shape[1]) for m in BLK))
common = sorted(set.intersection(*[set(b.index) for b in BLK.values()]))
emit("common samples: %d\n" % len(common))

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


def logL2(Xtr, ytr, Xte):
    return LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def block_cv(Atr, ytr, grp):
    ng = len(set(grp))
    if ng < 2 or Atr.shape[1] == 0:
        return 0.5
    oof = np.full(len(ytr), np.nan)
    for tri, vai in GroupKFold(min(3, ng)).split(Atr, ytr, grp):
        if len(set(ytr[tri])) < 2:
            continue
        try:
            oof[vai] = LinearSVC(C=0.02, class_weight="balanced", max_iter=3000).fit(Atr[tri], ytr[tri]).decision_function(Atr[vai])
        except Exception:
            pass
    a = auc(ytr, oof); return max(a, 1 - a) if a == a else 0.5


VARIANTS = ["naive", "zscore", "blocknorm", "pca", "residual", "drop", "mkl",
            "interact", "histgb", "rbfsvm", "mlp", "gate", "late", "hybrid"]
acc = {v: [] for v in VARIANTS}

for mut in MUTS:
    yall = D._labels_for_field_raw(ctx, mut).map(_m01); ym = D.labels_for_field(ctx, mut).map(_m01)
    tr = [s for s in common if pd.notna(ym.get(s)) and s not in hold]
    te = [s for s in common if s in hold and pd.notna(yall.get(s))]
    ytr = np.array([int(yall[s]) for s in tr]); yte = np.array([int(yall[s]) for s in te])
    if len(set(yte)) < 2 or len(set(ytr)) < 2:
        continue
    grp = dg.loc[tr].values
    raw_tr, raw_te = [], []                                 # naive: raw top-500
    z = {}; rel = {}                                        # z[mod]=(Atr,Ate) z-scored top-200 ; rel[mod]=cv auc
    for mod, B in BLK.items():
        Xtr = B.loc[tr].values; Xte = B.loc[te].values; keep = Xtr.std(0) > 0
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        if Xtr.shape[1] == 0:
            continue
        s1 = diff(Xtr, ytr, 500); raw_tr.append(Xtr[:, s1]); raw_te.append(Xte[:, s1])
        sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); Zte = sc.transform(Xte)
        s2 = diff(Ztr, ytr, 200); z[mod] = (Ztr[:, s2], Zte[:, s2]); rel[mod] = block_cv(Ztr[:, s2], ytr, grp)

    def cat(mods, scale=None):
        Atr = []; Ate = []
        for m in mods:
            if m not in z:
                continue
            a, b = z[m]
            if scale is not None:
                w = scale(m); a, b = a * w, b * w
            Atr.append(a); Ate.append(b)
        return (np.hstack(Atr), np.hstack(Ate)) if Atr else (None, None)

    def run(name, fn):
        try:
            s = fn()
            if s is not None:
                acc[name].append(auc(yte, s))
        except Exception:
            pass

    run("naive", lambda: logL2(np.hstack(raw_tr), ytr, np.hstack(raw_te)))
    run("zscore", lambda: (lambda ab: logL2(ab[0], ytr, ab[1]))(cat(z.keys())))
    run("blocknorm", lambda: (lambda ab: logL2(ab[0], ytr, ab[1]))(
        cat(z.keys(), scale=lambda m: 1.0 / (np.linalg.norm(z[m][0]) + 1e-9))))
    run("mkl", lambda: (lambda ab: logL2(ab[0], ytr, ab[1]))(cat(z.keys(), scale=lambda m: max(0.0, rel[m] - 0.5))))
    run("drop", lambda: (lambda ab: logL2(ab[0], ytr, ab[1]))(cat([m for m in z if m in INDEP])))
    run("gate", lambda: (lambda ab: logL2(ab[0], ytr, ab[1]))(cat([m for m in z if rel[m] > 0.55])) if any(rel[m] > 0.55 for m in z) else None)

    def pca_fn():
        Atr, Ate = [], []
        for m in z:
            a, b = z[m]; k = min(20, a.shape[1], len(tr) - 1)
            if k < 2:
                Atr.append(a); Ate.append(b); continue
            p = PCA(n_components=k, random_state=0).fit(a); Atr.append(p.transform(a)); Ate.append(p.transform(b))
        return logL2(np.hstack(Atr), ytr, np.hstack(Ate))
    run("pca", pca_fn)

    def residual_fn():
        if "RNA" not in z:
            return None
        Rtr, Rte = z["RNA"]; Atr, Ate = [Rtr], [Rte]
        for m in z:
            if m == "RNA":
                continue
            a, b = z[m]
            if m in IMPUTED:
                rg = Ridge(alpha=10.0).fit(Rtr, a); a = a - rg.predict(Rtr); b = b - rg.predict(Rte)
            Atr.append(a); Ate.append(b)
        return logL2(np.hstack(Atr), ytr, np.hstack(Ate))
    run("residual", residual_fn)

    def interact_fn():
        feats_tr, feats_te = [], []
        for m in z:
            a, b = z[m]; t = diff(a, ytr, 3)[:3]
            feats_tr.append(a[:, t]); feats_te.append(b[:, t])
        Ftr = np.hstack(feats_tr); Fte = np.hstack(feats_te)
        pt, pe = [Ftr], [Fte]
        for i in range(Ftr.shape[1]):
            for j in range(i + 1, Ftr.shape[1]):
                pt.append((Ftr[:, i] * Ftr[:, j])[:, None]); pe.append((Fte[:, i] * Fte[:, j])[:, None])
        return logL2(np.hstack(pt), ytr, np.hstack(pe))
    run("interact", interact_fn)

    run("histgb", lambda: (lambda ab: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
        l2_regularization=1.0, random_state=0).fit(ab[0], ytr).predict_proba(ab[1])[:, 1])(cat(z.keys())))
    run("rbfsvm", lambda: (lambda ab: SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
        probability=False).fit(ab[0], ytr).decision_function(ab[1]))(cat(z.keys())))
    run("mlp", lambda: (lambda ab: MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-2, max_iter=400,
        early_stopping=True, random_state=0).fit(ab[0], ytr).predict_proba(ab[1])[:, 1])(cat(z.keys())))

    # late fusion reference + hybrid (per-modality linSVM, reliability-weighted rank average)
    late_scores = None
    try:
        agg = np.zeros(len(te)); wsum = 0.0
        for m in z:
            a, b = z[m]
            sc = LinearSVC(C=0.02, class_weight="balanced", max_iter=4000).fit(a, ytr).decision_function(b)
            r = rankdata(sc) / len(sc); ca = auc(ytr, LinearSVC(C=0.02, class_weight="balanced", max_iter=4000).fit(a, ytr).decision_function(a))
            if ca < 0.5:
                r = 1 - r
            w = max(0.0, rel[m] - 0.5); agg += w * r; wsum += w
        if wsum > 0:
            late_scores = agg / wsum; acc["late"].append(auc(yte, late_scores))
    except Exception:
        pass
    try:
        ab = cat(z.keys()); early = logL2(ab[0], ytr, ab[1])
        if late_scores is not None:
            acc["hybrid"].append(auc(yte, 0.5 * (rankdata(early) / len(early)) + 0.5 * (rankdata(late_scores) / len(late_scores))))
    except Exception:
        pass

emit("%-11s %8s  (n mutations)" % ("variant", "mean AUC"))
order = sorted(VARIANTS, key=lambda v: -(np.mean(acc[v]) if acc[v] else -1))
for v in order:
    if acc[v]:
        emit("%-11s %8.3f   (%d)" % (v, np.mean(acc[v]), len(acc[v])))
emit("\nbaselines: naive early=%.3f | late=%.3f | (deployed linSVM+optimized late = 0.864)"
     % (np.mean(acc["naive"]) if acc["naive"] else np.nan, np.mean(acc["late"]) if acc["late"] else np.nan))
emit("\nEARLY IMPROVE OK")
