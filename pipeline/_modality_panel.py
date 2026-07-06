#!/usr/bin/env python3
"""Multi-modality panel x EVERY model (boss's directive + run-all-models).

Adds ADT / Lipid / Metabolite / GRN as feature blocks; feature-limited blocks kept WHOLE (no FS);
per-mutation SelectKBest (>=10 features/covariate, train-only) on the high-dim blocks (RNA, GRN).
Runs ALL models (not just the best linear ones), on raw AND batch-corrected ("corrected pseudobulk
folds") features, and reports the full model x modality-set grid with the WITH-RNA vs WITHOUT-RNA
comparison. Sealed-29 held-out, leakage-safe.

Modality SETS (composition = independent backbone in all multi-block sets):
  comp | RNA | ADT | Lipid | Metab | GRN | noRNA(comp+ADT+Lipid+Metab+GRN) | ALL
Models: logL2 logL1 elastic linSVM shrLDA PLS RF HistGB NaiveB kNN MLP(neural net)
-> runs/modality_panel/_results.txt
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

_NJ = int(os.environ.get("AMLMM_NJOBS", "4"))
ctx = build_context(Config(run_id="modality_panel"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]
    hold = set(ctx.holdout)
    ds = samples["dataset"].astype(str)

    # ---------- load every block ONCE ----------
    emit("loading modality blocks ...")
    comp = D._sample_level_matrix(ctx, "composition", set(samples.index))
    adt = dataio.sample_modality_matrix(ctx, "ADT")
    lip = dataio.sample_modality_matrix(ctx, "Lipid", min_spearman=0.3)
    met = dataio.sample_modality_matrix(ctx, "Metabolite", min_spearman=0.3)
    grn = dataio.sample_modality_matrix(ctx, "GRN")
    rmk = pio.udon_markers(ctx, "RNA")                      # cluster: UDON marker genes; local: empty
    rna = np.log1p(dataio.sample_modality_matrix(ctx, "RNA").clip(lower=0))
    if rmk:
        rna = rna[[g for g in rmk if g in rna.columns]]
        emit("  RNA: %d UDON markers" % rna.shape[1])
    else:
        _pool = [s for s in rna.index if s not in hold]
        _hvg = rna.loc[_pool].var(axis=0).sort_values(ascending=False).head(2000).index
        rna = rna[list(_hvg)]
        emit("  RNA: UDON markers absent -> top-2000 HVG on the %d-sample non-holdout pool" % len(_pool))
    BLOCKS = {"composition": comp, "ADT": adt, "Lipid": lip, "Metabolite": met, "GRN": grn, "RNA": rna}
    for k, v in BLOCKS.items():
        emit("  %-12s %5d feat  %4d samples" % (k, v.shape[1], v.shape[0]))
    HIGHDIM = {"GRN", "RNA"}
    common = sorted(set(comp.index) | set(adt.index) | set(grn.index) | set(rna.index))
    BLOCKS = {k: v.reindex(common) for k, v in BLOCKS.items()}

    SETS = {
        "comp":  ["composition"],
        "RNA":   ["composition", "RNA"],
        "ADT":   ["composition", "ADT"],
        "Lipid": ["composition", "Lipid"],
        "Metab": ["composition", "Metabolite"],
        "GRN":   ["composition", "GRN"],
        "noRNA": ["composition", "ADT", "Lipid", "Metabolite", "GRN"],
        "ALL":   ["composition", "ADT", "Lipid", "Metabolite", "GRN", "RNA"],
    }
    SET_ORDER = ["comp", "RNA", "ADT", "Lipid", "Metab", "GRN", "noRNA", "ALL"]

    # ---------- models (fixed hyperparams; tuning was ~neutral in the ablation) ----------
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
        ("MLP",     MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=500,
                                  early_stopping=True, random_state=0), "proba"),
    ]
    MODELS = [s[0] for s in SPECS]

    def fit_pred(spec, Xtr, ytr, Xte):
        name, base, kind = spec
        if kind == "pls":
            m = PLSRegression(n_components=min(10, Xtr.shape[1])).fit(Xtr, ytr.astype(float))
            return m.predict(Xte).ravel()
        est = clone(base)
        if kind == "sw":
            est.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            est.fit(Xtr, ytr)
        if kind == "dec":
            return est.decision_function(Xte)
        return est.predict_proba(Xte)[:, 1]

    # ---------- labels + testability gate ----------
    M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(common); inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    if os.environ.get("AMLMM_SMOKE"):                      # fast validation: 2 muts x 3 sets
        MUTS = MUTS[:2]; SET_ORDER = ["comp", "RNA", "noRNA"]
        store_sets = SET_ORDER
    emit("\ntestable mutations: %d | held-out with data: %d | models: %d | sets: %d\n"
         % (len(MUTS), sum(1 for s in hold if s in common), len(MODELS), len(SET_ORDER)))

    def _batch_correct(Xstd, train):
        """dataset-mean-center an already-standardized matrix using TRAIN means only (leakage-safe)."""
        Xc = Xstd.copy()
        gmean = Xstd.loc[train].mean(axis=0)
        for d in ds.reindex(Xstd.index).dropna().unique():
            trd = [s for s in train if ds.get(s) == d]
            if len(trd) >= 4:
                dm = Xstd.loc[trd].mean(axis=0)
                rows = [s for s in Xstd.index if ds.get(s) == d]
                Xc.loc[rows] = Xstd.loc[rows].values - dm.values + gmean.values
        return Xc

    def build_X(set_name, m, fs_k, bc):
        """Build (Xtr, Xte, ytr, yte) ONCE for a (set, mutation, preproc); models then share it.
        High-dim blocks -> per-mutation top-K (train-only); feature-limited blocks whole."""
        yall = D._labels_for_field_raw(ctx, m).map(_m01).reindex(common)
        ym = D.labels_for_field(ctx, m).map(_m01).reindex(common)
        train = [s for s in common if pd.notna(ym[s]) and s not in hold]
        test = [s for s in common if s in hold and pd.notna(yall[s])]
        yte = np.array([int(yall[s]) for s in test]); ytr = np.array([int(yall[s]) for s in train])
        if len(set(yte)) < 2 or len(set(ytr)) < 2:
            return None
        ptr, pte = [], []
        for blk in SETS[set_name]:
            B = BLOCKS[blk].fillna(0.0)
            sc = StandardScaler().fit(B.loc[train].values)
            Xall = pd.DataFrame(sc.transform(B.values), index=B.index, columns=B.columns)
            if bc:
                Xall = _batch_correct(Xall, train)
            Xtr = Xall.loc[train].values; Xte = Xall.loc[test].values
            if fs_k is not None and blk in HIGHDIM and Xtr.shape[1] > fs_k:
                skb = SelectKBest(f_classif, k=fs_k).fit(Xtr, ytr)
                Xtr, Xte = skb.transform(Xtr), skb.transform(Xte)
            ptr.append(Xtr); pte.append(Xte)
        return np.hstack(ptr), np.hstack(pte), ytr, yte

    # ============ run the grid: per (set, mut, preproc) build once, fit all models ============
    # store[(model, set, preproc)] = list of per-mutation AUCs
    store = {(mo, s, p): [] for mo in MODELS for s in SET_ORDER for p in ("raw", "bc")}
    permut = {(s, m): {} for s in SET_ORDER for m in MUTS}     # per-mutation best-model AUC (raw) for table 4
    for mi, m in enumerate(MUTS):
        for s in SET_ORDER:
            for p in ("raw", "bc"):
                built = build_X(s, m, fs_k=30, bc=(p == "bc"))
                if built is None:
                    continue
                Xtr, Xte, ytr, yte = built
                for spec in SPECS:
                    try:
                        au = roc_auc_score(yte, fit_pred(spec, Xtr, ytr, Xte))
                    except Exception:
                        au = np.nan
                    store[(spec[0], s, p)].append(au)
                    if p == "raw":
                        permut[(s, m)][spec[0]] = au
        emit("  [%2d/%2d] %s done" % (mi + 1, len(MUTS), m[:14]))

    def mean(mo, s, p):
        v = [a for a in store[(mo, s, p)] if a == a]
        return float(np.mean(v)) if v else float("nan")

    # ================= MATRIX A: raw features =================
    def render(preproc, title):
        emit("\n" + "=" * 104)
        emit("%s  (mean held-out AUC over %d mutations; high-dim RNA/GRN -> per-mutation top-30)" % (title, len(MUTS)))
        emit("=" * 104)
        emit("%-9s %s" % ("model", "  ".join("%-6s" % s for s in SET_ORDER)))
        for mo in MODELS:
            emit("%-9s %s" % (mo, "  ".join("%6.3f" % mean(mo, s, preproc) for s in SET_ORDER)))
        emit("%-9s %s" % ("--MEAN--", "  ".join(
            "%6.3f" % np.nanmean([mean(mo, s, preproc) for mo in MODELS]) for s in SET_ORDER)))
    render("raw", "MATRIX A  RAW FEATURES")
    render("bc", "MATRIX B  BATCH-CORRECTED FEATURES (corrected pseudobulk folds)")

    # ================= WITH vs WITHOUT RNA (the headline) =================
    emit("\n" + "=" * 78)
    emit("WITH-RNA vs WITHOUT-RNA per model  (each cell = best of raw/batch-corrected)")
    emit("=" * 78)
    emit("%-9s %8s %8s %8s   %10s %10s" % ("model", "comp", "RNA", "noRNA", "noRNA-RNA", "ALL"))
    def best(mo, s):
        return max(mean(mo, s, "raw"), mean(mo, s, "bc"))
    for mo in MODELS:
        cR, rR, nR, aR = best(mo, "comp"), best(mo, "RNA"), best(mo, "noRNA"), best(mo, "ALL")
        emit("%-9s %8.3f %8.3f %8.3f   %+10.3f %10.3f" % (mo, cR, rR, nR, nR - rR, aR))

    # overall winner
    cells = [(mo, s, p, mean(mo, s, p)) for mo in MODELS for s in SET_ORDER for p in ("raw", "bc")]
    cells = [c for c in cells if c[3] == c[3]]
    cells.sort(key=lambda c: -c[3])
    emit("\nTOP 10 (model x set x preproc):")
    for mo, s, p, v in cells[:10]:
        emit("  %-9s %-6s %-4s  %.3f" % (mo, s, p, v))

    # ================= per-mutation winning modality (best model per cell, raw) =================
    emit("\n" + "=" * 80)
    emit("PER-MUTATION best modality set (best model, raw)  -> which modality carries each mutation")
    emit("=" * 80)
    emit("%-12s %3s %8s %6s   %s" % ("mutation", "pos", "bestset", "AUC", "set ranking"))
    for m in MUTS:
        ylab = D._labels_for_field_raw(ctx, m).map(_m01).reindex(common)
        npos = int(sum(1 for s in common if s in hold and ylab.get(s) == 1.0))
        setbest = {}
        for s in SET_ORDER:
            vals = [v for v in permut[(s, m)].values() if v == v]
            if vals:
                setbest[s] = max(vals)
        if not setbest:
            continue
        order = sorted(setbest, key=lambda s: -setbest[s])
        emit("%-12s %3d %8s %6.2f   %s" % (m[:12], npos, order[0], setbest[order[0]],
             ", ".join("%s=%.2f" % (s, setbest[s]) for s in order[:5])))

    emit("\nMODALITY PANEL (all models) OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
