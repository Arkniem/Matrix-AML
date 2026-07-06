#!/usr/bin/env python3
"""Train the deployable MutationPredictor on the cohort + emit per-sample board reports.

Self-contained (no external OOF dependency): per (mutation, modality) fit StandardScaler -> differential
top-500 -> LinearSVC on that modality's own non-holdout labelled samples, AND run a donor-grouped 3-fold
CV to get honest OOF percentiles. Per mutation, optimised modality weights = ridge-NNLS of the per-modality
OOF percentiles (aligned on the common train samples) onto truth, floored to the best single modality and
falling back to a uniform mix of the informative (CV-AUC>0.5) modalities so NO mutation trains empty.
Persist -> mutation_predictor.pkl. Then score the sealed held-out (predicted vs known) + control gate and
write one patient_report.json per held-out sample for the board.  Run on an LSF compute node.
"""
import os, sys, json, pickle, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, genetics, udon_features as UF
from amlmm.predictor import MutationPredictor, ModalityModel, diff_select, _pct
try:
    import control_gate as CG
except Exception:
    CG = None

ctx = build_context(Config(run_id="single_modality"))
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.dirname(os.path.dirname(ctx.path("x")))     # <base>/runs
os.makedirs(RUNS_ROOT, exist_ok=True)
samples = ctx.tables["samples"]; dg = samples["donor_group"].astype(str); hold = set(ctx.holdout)
MODS = ["RNA", "Composition", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
def log(m): print(m, flush=True)


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
        log("skip %s: %s" % (m, e))
MODS = [m for m in MODS if m in BLK]
log("modalities: %s" % ", ".join(MODS))

M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
_m01 = {"present": 1.0, "absent": 0.0}
def auc(y, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if (ok.sum() >= 4 and len(set(y[ok])) == 2) else np.nan

allidx = set(BLK["RNA"].index)
MUTS = []
for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
    ym = D.labels_for_field(ctx, f).map(_m01)
    tr = [s for s in allidx if pd.notna(ym.get(s)) and s not in hold]
    yv = np.array([int(ym[s]) for s in tr])
    if (yv == 1).sum() >= 8 and (yv == 0).sum() >= 8:        # well-powered only
        MUTS.append(f)
log("trainable mutations: %d" % len(MUTS))

P = MutationPredictor(); P.modalities = MODS
P.mutations = [m.replace("mut_", "").replace("cyto_", "") for m in MUTS]
NAME = {m.replace("mut_", "").replace("cyto_", ""): m for m in MUTS}
heldout_pred = {}                                              # sample -> {mut_short: prob}


def cv_oof(B, ids, y, grp):
    """donor-grouped 3-fold OOF percentile scores for linSVM on this modality block."""
    X = B.loc[ids].values; keep = X.std(0) > 0
    if keep.sum() < 2 or len(set(grp)) < 2:
        return None
    oof = np.full(len(ids), np.nan)
    for tri, vai in GroupKFold(min(3, len(set(grp)))).split(X, y, grp):
        if len(set(y[tri])) < 2:
            continue
        sc = StandardScaler().fit(X[tri][:, keep]); Ztr = sc.transform(X[tri][:, keep]); Zva = sc.transform(X[vai][:, keep])
        sel = diff_select(Ztr, y[tri], 500)
        d = LinearSVC(C=0.02, class_weight="balanced", max_iter=4000).fit(Ztr[:, sel], y[tri]).decision_function(Zva[:, sel])
        oof[vai] = d
    ok = ~np.isnan(oof)
    if ok.sum() < 4:
        return None
    p = np.full(len(ids), np.nan); p[ok] = _pct(np.sort(oof[ok]), oof[ok])
    a = auc(y[ok], p[ok])
    if a == a and a < 0.5:
        p = 1 - p
    return p, (max(a, 1 - a) if a == a else 0.5)


for mflag in MUTS:
    mshort = mflag.replace("mut_", "").replace("cyto_", "")
    yall = D._labels_for_field_raw(ctx, mflag).map(_m01); ym = D.labels_for_field(ctx, mflag).map(_m01)
    oof_p, cvauc, te_p, mod_names = {}, {}, {}, []
    for mod in MODS:
        B = BLK[mod]
        tr = [s for s in B.index if pd.notna(ym.get(s)) and s not in hold]
        te = [s for s in B.index if s in hold and pd.notna(yall.get(s))]
        ytr = np.array([int(yall[s]) for s in tr])
        if (ytr == 1).sum() < 5 or (ytr == 0).sum() < 5:
            continue
        Xtr = B.loc[tr].values; keep = Xtr.std(0) > 0
        sc = StandardScaler().fit(Xtr[:, keep]); Ztr = sc.transform(Xtr[:, keep])
        sel = diff_select(Ztr, ytr, 500)
        svm = LinearSVC(C=0.02, class_weight="balanced", max_iter=5000).fit(Ztr[:, sel], ytr)
        dtr = svm.decision_function(Ztr[:, sel]); ssorted = np.sort(dtr)
        a = auc(ytr, _pct(ssorted, dtr)); sign = 1.0 if (a != a or a >= 0.5) else -1.0
        cv = cv_oof(B, tr, ytr, dg.loc[tr].values)
        P.models[(mshort, mod)] = ModalityModel(list(B.columns), sc, keep, sel, svm, ssorted, sign,
                                                round(cv[1], 3) if cv else None)
        mod_names.append(mod)
        if cv:
            oof_p[mod] = dict(zip(tr, cv[0])); cvauc[mod] = cv[1]
        if te:
            dte = svm.decision_function(sc.transform(B.loc[te].values[:, keep])[:, sel])
            q = _pct(ssorted, dte); te_p[mod] = dict(zip(te, q if sign > 0 else 1 - q))
    # optimised weights: NNLS of per-modality OOF percentiles (common train) onto truth ; floor + uniform fallback
    wm = {m: 0.0 for m in mod_names}
    common = sorted(set.intersection(*[set(oof_p[m]) for m in oof_p])) if oof_p else []
    yco = np.array([int(yall[s]) for s in common]) if common else np.array([])
    if len(common) >= 8 and len(set(yco)) == 2:
        cols = [m for m in mod_names if m in oof_p]
        O = np.column_stack([[oof_p[m].get(s, 0.5) for s in common] for m in cols])
        A = np.vstack([O, np.eye(O.shape[1])]); b = np.concatenate([yco.astype(float), np.zeros(O.shape[1])])
        w, _ = nnls(A, b)
        single = {m: auc(yco, np.array([oof_p[m][s] for s in common])) for m in cols}
        bs = max(single, key=lambda k: (single[k] if single[k] == single[k] else -1))
        comb = auc(yco, O @ w / (w.sum() or 1)) if w.sum() > 0 else np.nan
        if not (comb == comb and comb >= (single[bs] if single[bs] == single[bs] else -1) - 1e-9):
            w = np.zeros(len(cols)); w[cols.index(bs)] = 1.0
        tot = w.sum() or 1.0
        for m, wi in zip(cols, w):
            wm[m] = round(float(wi / tot), 3)
    if sum(wm.values()) == 0:                                  # fallback: uniform over informative modalities
        info = [m for m in mod_names if cvauc.get(m, 0) > 0.52] or mod_names
        for m in info:
            wm[m] = round(1.0 / len(info), 3)
    P.weights[mshort] = wm
    # OOF-calibrated present/absent threshold (Youden's J on the deployed OOF blend). Honest: out-of-fold
    # percentiles generalize, so it fixes the 0.5 over-calling WITHOUT the in-sample optimism that tanks
    # sensitivity. Falls back to 0.5 when the OOF is too small.
    P.thresholds[mshort] = 0.5
    if len(common) >= 8 and len(set(yco)) == 2:
        wc = [m for m in mod_names if m in oof_p and wm.get(m, 0) > 0]
        if wc:
            Oc = np.column_stack([[oof_p[m].get(s, 0.5) for s in common] for m in wc])
            wv = np.array([wm[m] for m in wc]); comb = Oc @ wv / (wv.sum() or 1.0)
            bt, bj = 0.5, -1.0
            for t in np.unique(comb):
                pr = comb >= t
                tp = int((pr & (yco == 1)).sum()); fn = int(((~pr) & (yco == 1)).sum())
                tn = int(((~pr) & (yco == 0)).sum()); fp = int((pr & (yco == 0)).sum())
                tpr = tp / (tp + fn) if (tp + fn) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                if tpr - fpr > bj:
                    bj, bt = tpr - fpr, float(t)
            P.thresholds[mshort] = round(bt, 3)
    P.prevalence[mshort] = round(float(np.mean([int(yall[s]) for s in allidx if pd.notna(ym.get(s)) and s not in hold])), 3)
    # held-out AUC of the deployed combiner
    te_all = sorted(set().union(*[set(te_p[m]) for m in te_p])) if te_p else []
    if te_all:
        yte = np.array([int(yall[s]) for s in te_all]); agg = np.full(len(te_all), np.nan)
        num = np.zeros(len(te_all)); den = np.zeros(len(te_all))
        for i, s in enumerate(te_all):
            for m in mod_names:
                if s in te_p.get(m, {}) and wm.get(m, 0) > 0:
                    num[i] += wm[m] * te_p[m][s]; den[i] += wm[m]
            if den[i] > 0:
                agg[i] = num[i] / den[i]; heldout_pred.setdefault(s, {})[mshort] = float(agg[i])
        ha = auc(yte, agg)
        P.heldout_auc[mshort] = round(float(ha), 3) if ha == ha else None
    log("  %-13s w=%s auc=%s" % (mshort, {k: v for k, v in wm.items() if v > 0}, P.heldout_auc.get(mshort)))

valid = [v for v in P.heldout_auc.values() if v is not None]
P.meta = {"rna": "raw+prog (UDON markers + program fractions)", "base_model": "linSVM",
          "n_train_samples": len([s for s in allidx if s not in hold]),
          "trained": time.strftime("%Y-%m-%d %H:%M")}
P.save(os.path.join(HERE, "mutation_predictor.pkl"))
log("\nsaved mutation_predictor.pkl | %d mutations, mean held-out AUC %.3f over %d"
    % (len(P.mutations), float(np.mean(valid)) if valid else float("nan"), len(valid)))

# ---- per-held-out-sample board reports (ALL sealed holdout, calibrated calls + abstention) ----
gate = CG.load_gate() if CG else None
NPOS = {m: int(sum(1 for s in allidx if s not in hold
                   and D._labels_for_field_raw(ctx, NAME[m]).map(_m01).get(s) == 1)) for m in P.mutations}
nrep = 0; no_data = []
for s in sorted(hold):
    sample = {mod: BLK[mod].loc[s] for mod in MODS if s in BLK[mod].index}
    preds = []
    for mshort in P.mutations:
        if not any((mshort, mod) in P.models for mod in MODS):
            continue
        pr = P.predict_one(mshort, sample); pr["mutation"] = mshort
        tl = D._labels_for_field_raw(ctx, NAME[mshort]).map(_m01).get(s)
        pr["true_label"] = ("present" if tl == 1 else "absent") if pd.notna(tl) else None
        pr["confidence"] = "abstain: underpowered (n+<3)" if NPOS.get(mshort, 0) < 3 else "ok"
        preds.append(pr)
    preds.sort(key=lambda p: -(p["probability"] or 0))
    spec = None
    if gate is not None and "Composition" in sample:
        comp = sample["Composition"]
        spec = CG.score_gate(gate, pd.Series(comp.values, index=comp.index))
    rep = {"mode": "mutation_panel", "sample_key": s, "dataset": s.split("::")[0] if "::" in s else None,
           "specimen_class": (spec["call"] if spec else None), "control_gate": spec,
           "mutation_predictions": preds, "predictor": P.summary(), "validation": True,
           "modalities_available": sorted(sample.keys()),
           "note": "sealed held-out sample — predicted vs known labels (OOF-calibrated thresholds)"}
    if not sample:
        rep["note"] = "held-out sample has no scRNA/atlas modality data — cannot be scored"; no_data.append(s)
    d = os.path.join(RUNS_ROOT, "predict_" + "".join(ch if ch.isalnum() else "_" for ch in s))
    os.makedirs(d, exist_ok=True)
    json.dump(rep, open(os.path.join(d, "patient_report.json"), "w"), default=str, indent=1)
    nrep += 1
log("wrote %d held-out reports (all %d holdout); no-data: %s" % (nrep, len(hold), no_data or "none"))
log("thresholds: median %.2f range %.2f-%.2f" % (
    float(np.median(list(P.thresholds.values()))) if P.thresholds else 0.5,
    min(P.thresholds.values()) if P.thresholds else 0.5, max(P.thresholds.values()) if P.thresholds else 0.5))
log("TRAIN PREDICTOR OK")
