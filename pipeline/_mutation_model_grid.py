#!/usr/bin/env python3
"""Full mutation x model AUC charts (OLD vs NEW vs ALL, FAIR) + improvement tests.

FAIRNESS FIXES vs the panel:
  * RNA normalized properly: CP10k + log1p on the full transcriptome, THEN subset to UDON markers
    (the panel used bare log1p -> missing library-size normalization, a real RNA handicap).
  * RNA kept WHOLE (all 385 markers, no per-mutation FS). Only GRN (7486) gets FS.

SETS: OLD=comp+RNA | NEW=comp+ADT+Lipid+Metab+GRN(FS) | ALL=old+new.
Charts: mutation(row) x model(col) held-out AUC for OLD/NEW/ALL (raw, sealed-29, leakage-safe).
Then: (1) GRN whole-vs-FS symmetry check; (2) per-mutation modality+model ROUTER (train-CV pick);
(3) top-model ENSEMBLE (mean-rank); (4) LATE FUSION (per-modality classifier, mean-rank) vs early.
-> runs/mutation_model_grid/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
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

_NJ = int(os.environ.get("AMLMM_NJOBS", "4"))
ctx = build_context(Config(run_id="mutation_model_grid"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]; hold = set(ctx.holdout); ds = samples["dataset"].astype(str)
    dg = samples["donor_group"].astype(str)
    emit("loading modality blocks ...")
    comp = D._sample_level_matrix(ctx, "composition", set(samples.index))
    adt = dataio.sample_modality_matrix(ctx, "ADT")
    lip = dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
    met = dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
    grn = dataio.sample_modality_matrix(ctx, "GRN")
    rmk = pio.udon_markers(ctx, "RNA")
    rna_full = dataio.sample_modality_matrix(ctx, "RNA")                    # raw mean-counts, full transcriptome
    rna_norm = pd.DataFrame(pio.cp10k_log1p(rna_full.values),               # CP10k + log1p (FIX)
                            index=rna_full.index, columns=rna_full.columns)
    if rmk:
        rna = rna_norm[[g for g in rmk if g in rna_norm.columns]]; emit("  RNA: CP10k+log1p, %d UDON markers (whole)" % rna.shape[1])
    else:
        _pool = [s for s in rna_norm.index if s not in hold]
        rna = rna_norm[list(rna_norm.loc[_pool].var(axis=0).sort_values(ascending=False).head(2000).index)]
        emit("  RNA: CP10k+log1p, top-2000 HVG (no UDON markers locally)")
    BLOCKS = {"composition": comp, "ADT": adt, "Lipid": lip, "Metabolite": met, "GRN": grn, "RNA": rna}
    for k, v in BLOCKS.items():
        emit("  %-12s %5d feat" % (k, v.shape[1]))
    common = sorted(set(comp.index) | set(adt.index) | set(grn.index) | set(rna.index))
    BLOCKS = {k: v.reindex(common) for k, v in BLOCKS.items()}

    SETS = {"OLD": ["composition", "RNA"],
            "NEW": ["composition", "ADT", "Lipid", "Metabolite", "GRN"],
            "ALL": ["composition", "ADT", "Lipid", "Metabolite", "GRN", "RNA"]}
    SET_ORDER = ["OLD", "NEW", "ALL"]
    FS_K = 30

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
    SPEC = {s[0]: s for s in SPECS}
    MODELS = [s[0] for s in SPECS]
    STRONG = ["logL2", "linSVM", "PLS", "shrLDA"]

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
    emit("\ntestable mutations: %d | models: %d\n" % (len(MUTS), len(MODELS)))

    def split(m):
        yall = D._labels_for_field_raw(ctx, m).map(_m01).reindex(common)
        ym = D.labels_for_field(ctx, m).map(_m01).reindex(common)
        train = [s for s in common if pd.notna(ym[s]) and s not in hold]
        test = [s for s in common if s in hold and pd.notna(yall[s])]
        return train, test, np.array([int(yall[s]) for s in train]), np.array([int(yall[s]) for s in test])

    def block_mat(blk, train, test, ytr, grn_fs=True):
        B = BLOCKS[blk].fillna(0.0)
        sc = StandardScaler().fit(B.loc[train].values)
        Xtr = sc.transform(B.loc[train].values); Xte = sc.transform(B.loc[test].values)
        if blk == "GRN" and grn_fs and Xtr.shape[1] > FS_K:
            skb = SelectKBest(f_classif, k=FS_K).fit(Xtr, ytr); Xtr, Xte = skb.transform(Xtr), skb.transform(Xte)
        return Xtr, Xte

    def build_X(set_name, train, test, ytr, grn_fs=True):
        ptr, pte = [], []
        for blk in SETS[set_name]:
            a, b = block_mat(blk, train, test, ytr, grn_fs); ptr.append(a); pte.append(b)
        return np.hstack(ptr), np.hstack(pte)

    def cv_auc(Xtr, ytr, groups, spec, k=3):
        ng = len(set(groups))
        if ng < 2:
            return np.nan
        aus = []
        for tri, vai in GroupKFold(min(k, ng)).split(Xtr, ytr, groups):
            if len(set(ytr[tri])) < 2 or len(set(ytr[vai])) < 2:
                continue
            try:
                aus.append(roc_auc_score(ytr[vai], fit_scores(spec, Xtr[tri], ytr[tri], Xtr[vai])))
            except Exception:
                pass
        return float(np.mean(aus)) if aus else np.nan

    # ---- main pass: held-out scores (all models) + train-CV (strong models, for the router) ----
    SC = {}; YT = {}; CV = {}
    for mi, m in enumerate(MUTS):
        train, test, ytr, yte = split(m)
        if len(set(yte)) < 2 or len(set(ytr)) < 2:
            continue
        grp = dg.reindex(train).values
        for s in SET_ORDER:
            Xtr, Xte = build_X(s, train, test, ytr, grn_fs=True)
            YT[(s, m)] = yte
            for spec in SPECS:
                try:
                    SC[(s, m, spec[0])] = fit_scores(spec, Xtr, ytr, Xte)
                except Exception:
                    SC[(s, m, spec[0])] = None
            for mo in STRONG:
                CV[(s, m, mo)] = cv_auc(Xtr, ytr, grp, SPEC[mo])
        emit("  [%2d/%2d] %s done" % (mi + 1, len(MUTS), m[:14]))

    def auc_of(s, m, mo):
        sc = SC.get((s, m, mo))
        return roc_auc_score(YT[(s, m)], sc) if sc is not None and (s, m) in YT and len(set(YT[(s, m)])) > 1 else np.nan

    def chart(setname, title):
        emit("\n" + "=" * 118)
        emit("%s -- held-out AUC: mutation (row) x model (col)" % title)
        emit("=" * 118)
        emit("%-13s %s  %5s" % ("mutation", "  ".join("%-6s" % mo for mo in MODELS), "BEST"))
        for m in MUTS:
            cells, vals = [], []
            for mo in MODELS:
                a = auc_of(setname, m, mo); vals.append(a)
                cells.append("%6.2f" % a if a == a else "   -- ")
            good = [v for v in vals if v == v]
            emit("%-13s %s  %5.2f" % (m[:13], "  ".join(cells), max(good) if good else float("nan")))
        emit("%-13s %s" % ("--MEAN--", "  ".join("%6.3f" % np.nanmean([auc_of(setname, m, mo) for m in MUTS]) for mo in MODELS)))

    chart("OLD", "CHART 1  OLD  (composition + RNA, full 385 markers, CP10k+log1p)")
    chart("NEW", "CHART 2  NEW  (composition + ADT + Lipid + Metabolite + GRN; no RNA)")
    chart("ALL", "CHART 3  ALL  (old + new)")

    emit("\n" + "=" * 60)
    emit("PER-MODEL mean AUC  OLD vs NEW vs ALL")
    emit("=" * 60)
    emit("%-9s %8s %8s %8s   %9s" % ("model", "OLD", "NEW", "ALL", "NEW-OLD"))
    for mo in MODELS:
        o = np.nanmean([auc_of("OLD", m, mo) for m in MUTS]); n = np.nanmean([auc_of("NEW", m, mo) for m in MUTS]); a = np.nanmean([auc_of("ALL", m, mo) for m in MUTS])
        emit("%-9s %8.3f %8.3f %8.3f   %+9.3f" % (mo, o, n, a, n - o))

    # ---- (1) GRN whole vs FS (symmetry) ----
    emit("\n" + "=" * 60)
    emit("(1) GRN whole vs FS-top30 -- strong models, NEW set, mean AUC")
    emit("=" * 60)
    emit("%-9s %10s %10s %8s" % ("model", "GRN-FS30", "GRN-whole", "delta"))
    for mo in STRONG:
        fs_a, wh_a = [], []
        for m in MUTS:
            train, test, ytr, yte = split(m)
            if len(set(yte)) < 2:
                continue
            try:
                Xtr, Xte = build_X("NEW", train, test, ytr, grn_fs=True);  fs_a.append(roc_auc_score(yte, fit_scores(SPEC[mo], Xtr, ytr, Xte)))
                Xtr, Xte = build_X("NEW", train, test, ytr, grn_fs=False); wh_a.append(roc_auc_score(yte, fit_scores(SPEC[mo], Xtr, ytr, Xte)))
            except Exception:
                pass
        emit("%-9s %10.3f %10.3f %+8.3f" % (mo, np.mean(fs_a), np.mean(wh_a), np.mean(wh_a) - np.mean(fs_a)))

    # ---- (2) per-mutation ROUTER: train-CV picks (set, model); report held-out ----
    emit("\n" + "=" * 80)
    emit("(2) PER-MUTATION ROUTER -- train-CV picks best (set,model); held-out AUC (no leakage)")
    emit("=" * 80)
    emit("%-13s %3s %8s %8s   %-16s %6s" % ("mutation", "pos", "OLDbest", "router", "picked", "cv"))
    router_aucs, single_best = [], []
    for m in MUTS:
        ylab = D._labels_for_field_raw(ctx, m).map(_m01).reindex(common)
        npos = int(sum(1 for s in common if s in hold and ylab.get(s) == 1.0))
        cand = [(s, mo, CV.get((s, m, mo), np.nan)) for s in SET_ORDER for mo in STRONG]
        cand = [c for c in cand if c[2] == c[2]]
        if not cand:
            continue
        bs, bmo, bcv = max(cand, key=lambda c: c[2])
        ho = auc_of(bs, m, bmo)
        oldbest = max([auc_of("OLD", m, mo) for mo in MODELS if auc_of("OLD", m, mo) == auc_of("OLD", m, mo)] or [np.nan])
        router_aucs.append(ho); single_best.append(oldbest)
        emit("%-13s %3d %8.2f %8.2f   %-16s %6.2f" % (m[:13], npos, oldbest, ho, "%s/%s" % (bs, bmo), bcv))
    emit("\nROUTER mean held-out AUC: %.3f   (vs OLD best-model mean %.3f)"
         % (np.nanmean(router_aucs), np.nanmean(single_best)))

    # ---- (3) ENSEMBLE: mean-rank of strong models, per set ----
    emit("\n" + "=" * 50)
    emit("(3) ENSEMBLE (mean-rank of %s)" % "+".join(STRONG))
    emit("=" * 50)
    emit("%-8s %10s %12s" % ("set", "best-single", "ensemble"))
    for s in SET_ORDER:
        best_single = max(np.nanmean([auc_of(s, m, mo) for m in MUTS]) for mo in MODELS)
        ens = []
        for m in MUTS:
            if (s, m) not in YT or len(set(YT[(s, m)])) < 2:
                continue
            R = [rankdata(SC[(s, m, mo)]) for mo in STRONG if SC.get((s, m, mo)) is not None]
            if R:
                ens.append(roc_auc_score(YT[(s, m)], np.mean(R, axis=0)))
        emit("%-8s %10.3f %12.3f" % (s, best_single, np.nanmean(ens)))

    # ---- (4) LATE FUSION: per-modality logL2, mean-rank across modalities ----
    emit("\n" + "=" * 64)
    emit("(4) LATE FUSION (per-modality logL2, mean-rank) vs EARLY (concat)")
    emit("=" * 64)
    BSC = {}
    for m in MUTS:
        train, test, ytr, yte = split(m)
        if len(set(yte)) < 2:
            continue
        for blk in ["composition", "ADT", "Lipid", "Metabolite", "GRN", "RNA"]:
            try:
                Xtr, Xte = block_mat(blk, train, test, ytr, grn_fs=True)
                BSC[(blk, m)] = fit_scores(SPEC["logL2"], Xtr, ytr, Xte)
            except Exception:
                BSC[(blk, m)] = None
    for name, blks, early_set in [("noRNA", ["composition", "ADT", "Lipid", "Metabolite", "GRN"], "NEW"),
                                  ("ALL", ["composition", "ADT", "Lipid", "Metabolite", "GRN", "RNA"], "ALL")]:
        fus, early = [], []
        for m in MUTS:
            if (early_set, m) not in YT or len(set(YT[(early_set, m)])) < 2:
                continue
            R = [rankdata(BSC[(b, m)]) for b in blks if BSC.get((b, m)) is not None]
            if R:
                fus.append(roc_auc_score(YT[(early_set, m)], np.mean(R, axis=0)))
            early.append(auc_of(early_set, m, "logL2"))
        emit("  %-6s  late-fusion %.3f   early-fusion(logL2) %.3f" % (name, np.nanmean(fus), np.nanmean(early)))

    emit("\nMUTATION-MODEL GRID + IMPROVEMENTS OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
