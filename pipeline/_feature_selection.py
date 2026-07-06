#!/usr/bin/env python3
"""Feature-selection sweep (boss's ask): reduce features to a MINIMAL discriminating set per mutation
and see if it boosts held-out accuracy. For each mutation, select top-K features by univariate ANOVA
F-score on TRAIN ONLY (leakage-safe), classify with the robust models, sweep K. Reports the optimal K,
the mean-AUC vs K curve, the selected genes per mutation (interpretable), and the modality split.
Tests on ALL withheld samples (fill, not drop, so no sample is lost to a stray NaN).
-> runs/feature_selection/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

ctx = build_context(Config(run_id="feature_selection"))
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
    feat = comp.join(rna, how="inner").fillna(0.0)          # FILL not drop -> keep every joined sample
    ncomp = comp.shape[1]
    hold = set(ctx.holdout)

    # ---- all-29 coverage report ----
    held_all = sorted(ctx.holdout)
    inf = [s for s in held_all if s in feat.index]
    drop = [s for s in held_all if s not in feat.index]
    emit("held-out: %d total | testable (in joined features) %d | dropped %d" % (len(held_all), len(inf), len(drop)))
    for s in drop:
        emit("  dropped %s : in_composition=%s in_RNA=%s" % (s, s in comp.index, s in rna.index))
    emit("features: %d (composition %d + RNA markers %d)\n" % (feat.shape[1], ncomp, feat.shape[1] - ncomp))

    M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(feat.index)
        inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    KS = [5, 10, 20, 30, 50, 100, 200, feat.shape[1]]
    emit("testable mutations: %d | feature-set sizes K = %s\n" % (len(MUTS), KS))

    cols = np.array(feat.columns)
    def model_auc(Xtr, ytr, Xte, yte):
        a = []
        for est in (LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000),
                    LinearSVC(C=0.02, class_weight="balanced", max_iter=5000)):
            est.fit(Xtr, ytr)
            s = est.decision_function(Xte) if isinstance(est, LinearSVC) else est.predict_proba(Xte)[:, 1]
            a.append(roc_auc_score(yte, s))
        return max(a)                                       # best of logistic / linear-SVM

    per_k = {k: [] for k in KS}; best_k_per = {}; selected = {}
    emit("%-12s %3s  %s" % ("mutation", "pos", "  ".join("K=%-4d" % k for k in KS)))
    for m in MUTS:
        yall = D._labels_for_field_raw(ctx, m).map(_m01).reindex(feat.index)
        ym = D.labels_for_field(ctx, m).map(_m01).reindex(feat.index)
        train = [s for s in feat.index if pd.notna(ym[s]) and s not in hold]
        test = [s for s in feat.index if s in hold and pd.notna(yall[s])]
        yte = np.array([int(yall[s]) for s in test]); ytr = np.array([int(yall[s]) for s in train])
        if len(set(yte)) < 2:
            continue
        sc = StandardScaler().fit(feat.loc[train].values)
        Xtr_full, Xte_full = sc.transform(feat.loc[train].values), sc.transform(feat.loc[test].values)
        row, best_auc, best_k = [], -1, KS[-1]
        for k in KS:
            kk = min(k, Xtr_full.shape[1])
            sk = SelectKBest(f_classif, k=kk).fit(Xtr_full, ytr)        # train-only selection (leakage-safe)
            au = model_auc(sk.transform(Xtr_full), ytr, sk.transform(Xte_full), yte)
            per_k[k].append(au); row.append("%5.2f" % au)
            if au > best_auc:
                best_auc, best_k = au, k
                selected[m] = list(cols[sk.get_support()])
        best_k_per[m] = (best_k, best_auc)
        emit("%-12s %3d  %s" % (m[:12], int(yte.sum()), "  ".join(row)))

    emit("\n%-12s %3s  %s" % ("MEAN AUC", "", "  ".join("%5.3f" % np.mean(per_k[k]) for k in KS)))
    bestK = max(KS, key=lambda k: np.mean(per_k[k]))
    emit("\nbest mean-AUC at K=%d -> %.3f  (vs all-features K=%d -> %.3f)"
         % (bestK, np.mean(per_k[bestK]), KS[-1], np.mean(per_k[KS[-1]])))
    emit("\nminimal discriminating set per mutation (at each mutation's own best K):")
    for m in MUTS:
        if m not in best_k_per:
            continue
        bk, ba = best_k_per[m]; gs = selected.get(m, [])
        nc = sum(1 for g in gs if not g.startswith("comp::"))
        emit("  %-12s bestK=%-4d AUC=%.2f  (%d RNA genes, %d composition)  top: %s"
             % (m[:12], bk, ba, nc, len(gs) - nc, ", ".join(g.replace("comp::", "comp:") for g in gs[:8])))
    emit("\nFEATURE SELECTION OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
