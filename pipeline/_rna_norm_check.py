#!/usr/bin/env python3
"""Settle 'what happened to 0.84': did CP10k normalization break RNA, or did it remove a
library-size/batch artifact? Re-run OLD = composition + RNA(385 markers) under TWO normalizations
in the identical framework, all 11 models, sealed-29 held-out. Plus diagnostics:
  * total-count-ALONE AUC per mutation (does library size alone predict the mutation -> leakage?)
  * eta^2 of log-total across datasets (is library size confounded with cohort/batch?)
-> runs/rna_norm_check/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
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

_NJ = int(os.environ.get("AMLMM_NJOBS", "4"))
ctx = build_context(Config(run_id="rna_norm_check"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]; hold = set(ctx.holdout); ds = samples["dataset"].astype(str)
    comp = D._sample_level_matrix(ctx, "composition", set(samples.index))
    rmk = pio.udon_markers(ctx, "RNA")
    rna_full = dataio.sample_modality_matrix(ctx, "RNA")
    common = sorted(set(comp.index) | set(rna_full.index))
    comp = comp.reindex(common)
    rna_full = rna_full.reindex(common)
    mk = [g for g in rmk if g in rna_full.columns] if rmk else \
        list(rna_full.loc[[s for s in common if s not in hold]].var().sort_values(ascending=False).head(2000).index)
    # two normalizations of the SAME marker block
    bare = np.log1p(rna_full[mk].clip(lower=0))
    cp10k = pd.DataFrame(pio.cp10k_log1p(rna_full.values), index=rna_full.index, columns=rna_full.columns)[mk]
    total = np.log1p(rna_full.fillna(0.0).sum(axis=1) + 1.0)          # per-sample library size
    emit("markers: %d | %s | held-out with data: %d"
         % (len(mk), "UDON" if rmk else "HVG2000", sum(1 for s in hold if s in common)))

    # library-size confound: eta^2 of log-total across datasets
    tv = total.reindex(common).dropna()
    g = ds.reindex(tv.index)
    grand = tv.mean()
    ssb = sum(len(tv[g == d]) * (tv[g == d].mean() - grand) ** 2 for d in g.unique())
    sst = float(((tv - grand) ** 2).sum())
    emit("library-size (log-total) eta^2 across %d datasets: %.3f  (1.0 = total is fully a cohort/batch label)\n"
         % (g.nunique(), ssb / sst if sst else float("nan")))

    SPECS = [
        ("logL2",   LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000), "proba"),
        ("logL1",   LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=3000), "proba"),
        ("elastic", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.1, class_weight="balanced", max_iter=4000), "proba"),
        ("linSVM",  LinearSVC(C=0.02, class_weight="balanced", max_iter=5000), "dec"),
        ("shrLDA",  LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "proba"),
        ("PLS",     "pls", "pls"),
        ("RF",      RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=0, n_jobs=_NJ), "proba"),
        ("HistGB",  HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0), "sw"),
        ("NaiveB",  GaussianNB(), "proba"),
        ("kNN",     KNeighborsClassifier(n_neighbors=15), "proba"),
        ("MLP",     MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0), "proba"),
    ]
    MODELS = [s[0] for s in SPECS]

    def fit_scores(spec, Xtr, ytr, Xte):
        name, base, kind = spec
        if kind == "pls":
            m = PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float)); return m.predict(Xte).ravel()
        est = clone(base)
        if kind == "sw":
            est.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            est.fit(Xtr, ytr)
        return est.decision_function(Xte) if kind == "dec" else est.predict_proba(Xte)[:, 1]

    M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(common); inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    emit("testable mutations: %d\n" % len(MUTS))

    def matrices(rna_block, m):
        yall = D._labels_for_field_raw(ctx, m).map(_m01).reindex(common)
        ym = D.labels_for_field(ctx, m).map(_m01).reindex(common)
        train = [s for s in common if pd.notna(ym[s]) and s not in hold]
        test = [s for s in common if s in hold and pd.notna(yall[s])]
        ytr = np.array([int(yall[s]) for s in train]); yte = np.array([int(yall[s]) for s in test])
        if len(set(yte)) < 2 or len(set(ytr)) < 2:
            return None
        feat = comp.join(rna_block, how="left").fillna(0.0)
        sc = StandardScaler().fit(feat.loc[train].values)
        return sc.transform(feat.loc[train].values), sc.transform(feat.loc[test].values), ytr, yte, train, test

    # held-out AUC per (norm, mutation, model) + total-alone
    AUC = {"bare": {}, "cp10k": {}}
    tot_alone = {}
    for m in MUTS:
        for nm, blk in [("bare", bare), ("cp10k", cp10k)]:
            built = matrices(blk, m)
            if built is None:
                continue
            Xtr, Xte, ytr, yte, train, test = built
            AUC[nm][m] = {}
            for spec in SPECS:
                try:
                    AUC[nm][m][spec[0]] = roc_auc_score(yte, fit_scores(spec, Xtr, ytr, Xte))
                except Exception:
                    AUC[nm][m][spec[0]] = np.nan
            if nm == "bare":
                ttr = total.reindex(train).values; tte = total.reindex(test).values
                # orient so AUC>=.5 (total-alone discriminative power, direction-agnostic)
                a = roc_auc_score(yte, tte); tot_alone[m] = max(a, 1 - a)
        emit("  %s done" % m[:14])

    def chart(nm, title):
        emit("\n" + "=" * 118)
        emit("%s -- held-out AUC: mutation (row) x model (col)" % title)
        emit("=" * 118)
        emit("%-13s %s  %5s %6s" % ("mutation", "  ".join("%-6s" % mo for mo in MODELS), "BEST", "totaln"))
        for m in MUTS:
            if m not in AUC[nm]:
                continue
            r = AUC[nm][m]
            cells = "  ".join("%6.2f" % r[mo] if r.get(mo) == r.get(mo) else "   -- " for mo in MODELS)
            vals = [v for v in r.values() if v == v]
            emit("%-13s %s  %5.2f %6.2f" % (m[:13], cells, max(vals) if vals else float("nan"), tot_alone.get(m, float("nan"))))
        emit("%-13s %s" % ("--MEAN--", "  ".join(
            "%6.3f" % np.nanmean([AUC[nm][m].get(mo, np.nan) for m in MUTS if m in AUC[nm]]) for mo in MODELS)))

    chart("bare", "OLD bare-log1p  (composition + RNA, NO library-size normalization)")
    chart("cp10k", "OLD CP10k+log1p  (composition + RNA, library-size normalized)")

    emit("\n" + "=" * 64)
    emit("PER-MODEL: bare-log1p vs CP10k  (what CP10k did to RNA)")
    emit("=" * 64)
    emit("%-9s %10s %10s %9s" % ("model", "bare", "CP10k", "CP10k-bare"))
    for mo in MODELS:
        b = np.nanmean([AUC["bare"][m].get(mo, np.nan) for m in MUTS if m in AUC["bare"]])
        c = np.nanmean([AUC["cp10k"][m].get(mo, np.nan) for m in MUTS if m in AUC["cp10k"]])
        emit("%-9s %10.3f %10.3f %+9.3f" % (mo, b, c, c - b))
    emit("\ntotal-count-ALONE mean AUC: %.3f  (high -> bare-log1p's edge is largely library size)"
         % np.nanmean(list(tot_alone.values())))
    emit("\nRNA NORM CHECK OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
