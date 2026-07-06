#!/usr/bin/env python3
"""Single-modality ablation for ONE modality (set by env AMLMM_MODALITY) -- run 8 in parallel.
Held-out AUC when this modality is the ONLY input, for every mutation x every model.
Loads ONLY this modality's block (light memory). Sealed-29, leakage-safe per-mutation FS.
-> runs/single_modality/auc_<MOD>.tsv  +  _results_<MOD>.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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

MOD = os.environ.get("AMLMM_MODALITY", "Composition")
_NJ = int(os.environ.get("AMLMM_NJOBS", "4"))
ctx = build_context(Config(run_id="single_modality"))
RES = ctx.path("_results_%s.txt" % MOD); open(RES, "w").close()
LONG = ctx.path("auc_%s.tsv" % MOD)
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]; hold = set(ctx.holdout)
    universe = sorted(samples.index)
    emit("modality: %s | loading its block only ..." % MOD)

    def load_block(mod):
        if mod == "Composition":
            return D._sample_level_matrix(ctx, "composition", set(samples.index))
        if mod == "RNA":
            r = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0))   # bare log1p (best)
            mk = pio.udon_markers(ctx, "RNA")
            if mk:
                return r[[g for g in mk if g in r.columns]]
            pool = [s for s in r.index if s not in hold]
            return r[list(r.loc[pool].var().sort_values(ascending=False).head(2000).index)]
        if mod == "ADT":
            return dataio.sample_modality_matrix(ctx, "ADT")
        if mod == "Lipid":
            return dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
        if mod == "Metabolite":
            return dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
        if mod == "GRN":
            return dataio.sample_modality_matrix(ctx, "GRN")
        if mod == "LSC":
            t = ctx.tables.get("lsc_calls")
            cols = [c for c in ["Prob_m-LSC", "Prob_p+m-LSC", "Prob_p-LSC", "MaxProb"] if t is not None and c in t.columns]
            return t[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=[])
        if mod == "Cell-comm":
            return dataio.cellcomm_matrix(ctx)
        raise SystemExit("unknown modality %s" % mod)

    import gc
    _cache = ctx.path("_sl_%s.pkl" % MOD)                  # disk cache of the aggregated sample-level matrix
    if os.path.exists(_cache):                             # memory-lean path: skip the multi-GB AnnData load+aggregation
        B = pd.read_pickle(_cache)
        emit("  %s: loaded cached sample-level matrix (%d features, %d samples)" % (MOD, B.shape[1], B.shape[0]))
    else:
        B = load_block(MOD)
        if B.shape[1] == 0:
            emit("[FAILED] modality %s produced 0 features" % MOD); raise SystemExit
        B = B[~B.index.duplicated(keep="first")]
        for _k in [k for k in list(ctx.tables) if k.startswith("_mem::") or k == "_cellcomm_ad"]:
            ctx.tables.pop(_k, None)
        gc.collect()
        try:
            B.to_pickle(_cache)                            # so the next run avoids the OOM-prone aggregation
        except Exception as _e:
            emit("  (cache save failed: %s)" % _e)
        emit("  %s: %d features | %d samples | %d/%d held-out covered"
             % (MOD, B.shape[1], B.shape[0], sum(1 for s in hold if s in set(B.index)), len(hold)))
    FULL_GRN = {"logL2", "linSVM", "PLS", "RF", "NaiveB", "kNN"}   # feasible on full 7486 from the disk cache
    def fs_k_for(model):
        # full features win on GRN (comparison: 0.756 vs 0.747), and the cache makes full feasible for the
        # fast models; only the 3 heavy ones (shrLDA cov, elastic-saga, MLP) plus liblinear-L1/HistGB still need FS.
        if MOD == "GRN":
            return None if model in FULL_GRN else 500     # 6 fast models: full 7486; the heavy 5: differential 500/side
        if MOD == "Cell-comm":
            return 500                                     # 141k full is infeasible -> differential 500/side for all
        return None                                        # every other modality: full features for every model

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
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(universe); inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    emit("  testable mutations: %d | models: %d\n" % (len(MUTS), len(MODELS)))

    Bf = B.fillna(0.0)
    rows = []
    grid = {m: {} for m in MUTS}
    for m in MUTS:
        yall = D._labels_for_field_raw(ctx, m).map(_m01)
        ym = D.labels_for_field(ctx, m).map(_m01)
        train = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
        test = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
        ytr = np.array([int(yall[s]) for s in train]); yte = np.array([int(yall[s]) for s in test])
        if len(set(yte)) < 2 or len(set(ytr)) < 2:
            continue
        Xtr0 = Bf.loc[train].values; Xte0 = Bf.loc[test].values
        keep = Xtr0.std(axis=0) > 0
        if keep.sum() == 0:
            continue
        Xtr0, Xte0 = Xtr0[:, keep], Xte0[:, keep]
        sc = StandardScaler().fit(Xtr0)
        Xtr_w, Xte_w = sc.transform(Xtr0), sc.transform(Xte0)      # full (whole) standardized matrices
        fs_cache = {None: (Xtr_w, Xte_w)}                          # per-K selected views, built once and reused
        def getX(kside):                                          # DIFFERENTIAL FS: top-kside UP + top-kside DOWN
            if kside is None or Xtr_w.shape[1] <= 2 * kside:
                return Xtr_w, Xte_w
            if kside not in fs_cache:
                F = np.nan_to_num(f_classif(Xtr_w, ytr)[0])        # association magnitude per feature
                md = Xtr_w[ytr == 1].mean(axis=0) - Xtr_w[ytr == 0].mean(axis=0)   # signed direction (mutant - wt)
                order = np.argsort(np.sign(md) * F)                # ascending: most-DOWN ... most-UP
                sel = np.unique(np.concatenate([order[:kside], order[-kside:]]))   # both tails
                fs_cache[kside] = (Xtr_w[:, sel], Xte_w[:, sel])
            return fs_cache[kside]
        npos = int(yte.sum())
        for spec in SPECS:
            Xtr, Xte = getX(fs_k_for(spec[0]))                     # differential 500/side on GRN & Cell-comm; full elsewhere
            try:
                au = roc_auc_score(yte, fit_scores(spec, Xtr, ytr, Xte))
            except Exception:
                au = np.nan
            grid[m][spec[0]] = au
            rows.append((MOD, m, spec[0], npos, au))
        emit("  %-13s pos=%2d done" % (m[:13], npos))

    pd.DataFrame(rows, columns=["modality", "mutation", "model", "npos", "auc"]).to_csv(LONG, sep="\t", index=False)
    emit("\nwrote %d rows -> %s" % (len(rows), LONG))
    emit("\n%-13s %s" % ("[%s] model" % MOD, "  ".join("%-6s" % mo for mo in MODELS)))
    emit("%-13s %s" % ("mean AUC", "  ".join("%6.3f" % np.nanmean([grid[m].get(mo, np.nan) for m in MUTS]) for mo in MODELS)))
    emit("\nSINGLE MODALITY [%s] OK" % MOD)
except SystemExit:
    raise
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
