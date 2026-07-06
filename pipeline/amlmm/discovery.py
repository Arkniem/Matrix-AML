"""Discovery agent — pseudobulk-level, per-(metadata-field x modality x cell-state) weight learning.

For each (field F, modality M, cell-state C): rows = the pseudobulks of state C whose sample has a
non-NA, usable F label; features = M's per-pseudobulk vectors; donor-grouped 5-fold CV via
`cv.nested_cv_evaluate` (within a cell-state there is one pseudobulk per sample, so donor grouping is
clean). The held-out balanced accuracy vs the permutation chance ceiling -> a permutation-CALIBRATED
weight; OOF per-pseudobulk predictions are the Deploy association seam; `cv.validated_markers` are the
discriminating features that ALSO separated on held-out. Sample-level modalities (composition,
cell-communication, LSC) run with no cell-state axis under the sentinel cell_state="__sample__".

This module reuses the credibility core unchanged (cv/models/targets/genetics); it adds orchestration,
the permutation->weight mapping, and the weight/OOF/marker harvest. The baseline path is untouched.
"""
from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass, field as _dc_field, asdict
from functools import partial

import numpy as np
import pandas as pd

from . import cv, models, targets, genetics, dataio, pseudobulk_io as pio
from .dataio import IMPUTED_MODALITIES

SAMPLE_LEVEL = ("composition", "cell-communication", "LSC")
PSEUDOBULK_MODALITIES = ("RNA", "GRN", "Metabolite", "Lipid", "ADT")
SAMPLE_SENTINEL = "__sample__"


@dataclass
class DiscoveryConfig:
    # skip-gates (degrade gracefully, never fake)
    min_pseudobulks: int = 20
    min_donors: int = 8
    min_class_n: int = 8
    min_donors_per_class: int = 2
    # significance + weight
    alpha: float = 0.05
    screen_permutations: int = 30            # pass-1 coarse permutation count (fail-fast)
    final_permutations: int = 200            # pass-2 precise count (only for combos that survive screen)
    screen_promote_alpha: float = 0.10       # promote screen->confirm iff screen perm_p <= this
                                             #   (loose: 30 perms underestimate near the boundary;
                                             #    promote near-misses too so the precise pass can settle them)
    # modeling knobs (swappable)
    prefilter_features: int = 800            # UNSUPERVISED top-variance cap applied BEFORE CV. Kept small
                                             #   because the permutation null's reference logreg sees this
                                             #   width and max_iter=5000 does NOT converge at p>>n (35.7k
                                             #   RNA genes / n~45) -> all 5000 iters -> ~100x slower.
    max_features: int = 300                  # in-fold SUPERVISED SelectKBest cap (leakage-safe; p~n)
    feature_selector: str = "f_classif"      # f_classif (fast ANOVA, default) | mutual_info (slow)
    models: tuple = ("rf", "logreg")
    normalization: dict = _dc_field(default_factory=lambda: {"RNA": "cp10k_log1p"})
    strategy: str = "donor_kfold"
    marker_k: int = 15


def _fidelity_tag(modality) -> str:
    if modality in ("composition", "RNA", "cell-communication"):
        return "measured"
    if modality in ("Metabolite", "Lipid"):
        return "fidelity_filtered"   # var has heldout_spearman -> filtered >=0.3
    if modality in ("GRN", "ADT"):
        return "unknown"             # no held-out fidelity metric (do not fabricate)
    if modality == "LSC":
        return "classifier"
    return "unknown"


def labels_for_field(ctx, field) -> pd.Series:
    """sample_key -> label for a metadata field, with HELD-OUT test samples masked to NaN so they are
    excluded from Discovery training (and therefore from every training combo + Deploy reference set).
    `ctx.holdout` is loaded from pipeline/holdout_samples.txt by build_context (empty -> no masking)."""
    lab = _labels_for_field_raw(ctx, field)
    ho = getattr(ctx, "holdout", None)
    if ho:
        lab = lab.copy()
        lab[lab.index.isin(ho)] = np.nan
    return lab


def _labels_for_field_raw(ctx, field) -> pd.Series:
    """sample_key -> label for a metadata field. TARGETS fields via targets.get_labels (subtype uses
    the canonical driver map); 'mut_*'/'cyto_*' genetic flag columns via genetics.build_mutation_matrix
    as binary present/absent (the headline 'which mutations predictable by which modality' axis).
    A flag is 'absent' only for samples that HAVE genetic data; otherwise NA (unknown, not negative)."""
    if field in targets.TARGETS:
        return targets.get_labels(ctx, field)
    M = ctx.tables.get("mutations")
    if M is None:
        M = genetics.build_mutation_matrix(ctx)
    if field in M.columns:
        col = M[field]
        has = (M["has_genetic_data"].astype(bool) if "has_genetic_data" in M.columns
               else pd.Series(True, index=M.index))
        lab = pd.Series(np.where(col == 1.0, "present", "absent"), index=M.index, dtype=object)
        lab[(~has) & (col != 1.0)] = np.nan
        return lab
    return pd.Series(index=ctx.tables["samples"].index, dtype=object)


def candidate_fields(ctx) -> list:
    """Metadata fields Discovery can attempt: TARGETS multiclass/binary + present mutation/cyto flags."""
    fields = [t for t, spec in targets.TARGETS.items() if spec["kind"] in ("multiclass", "binary")]
    M = ctx.tables.get("mutations")
    if M is None:
        M = genetics.build_mutation_matrix(ctx)
    flags = [c for c in M.columns if c.startswith(("mut_", "cyto_"))]
    return fields + flags


# --------------------------------------------------------------------------- feature matrices
def _sample_level_matrix(ctx, modality, sample_keys) -> pd.DataFrame:
    """A sample-level feature block (no cell-state axis), indexed by sample_key."""
    sk = [k for k in sample_keys]
    if modality == "composition":
        comp = ctx.tables["composition"]
        comp = comp.reindex([k for k in sk if k in comp.index])
        comp = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        comp.columns = [f"comp::{c}" for c in comp.columns]
        return comp
    if modality == "cell-communication":
        return dataio.cellcomm_matrix(ctx, sk)
    if modality == "LSC":
        lsc = ctx.tables.get("lsc_calls")
        if lsc is None:
            return pd.DataFrame(index=[])
        prob = lsc[[c for c in lsc.columns if c.startswith("Prob_")]].apply(pd.to_numeric, errors="coerce")
        return prob.reindex([k for k in sk if k in prob.index]).dropna(how="all")
    return pd.DataFrame(index=[])


def _combo_frame(ctx, modality, cell_state, sample_keys):
    """Return (X, sample_key[], donor[], cohort[]) aligned by row. X rows = pseudobulk ids
    (pseudobulk modality) or sample_keys (sample-level modality)."""
    s = ctx.tables["samples"]
    if modality in SAMPLE_LEVEL:
        X = _sample_level_matrix(ctx, modality, sample_keys)
        if X.shape[0] == 0:
            return X, None, None, None
        skv = np.array(list(X.index))
    else:
        ms = 0.3 if modality in IMPUTED_MODALITIES else None
        X = pio.cellstate_modality_matrix(ctx, modality, cell_state, sample_keys=sample_keys, min_spearman=ms)
        if X.shape[0] == 0:
            return X, None, None, None
        skv = ctx.tables["pseudobulks"].loc[X.index, "sample_key"].values
    donor = s.reindex(skv)["donor_group"].astype(str).values
    cohort = s.reindex(skv)["dataset"].astype(str).values
    return X, skv, donor, cohort


def _factories(modality, config, n_features, selector=None, k=None) -> dict:
    """Model pipelines with modality-appropriate in-fold preprocessing prepended (leakage-safe:
    fit per fold). RNA -> CP10k/log1p; high-dim -> supervised SelectKBest. `selector`/`k` default to
    the config (the `discovery_feature_select` hook may override per combo); selector="none" disables
    supervised selection (the unsupervised variance prefilter still applies upstream)."""
    from sklearn.pipeline import Pipeline
    selector = config.feature_selector if selector is None else selector
    k = int(config.max_features if k is None else k)
    pre = []
    if config.normalization.get(modality, "none") == "cp10k_log1p":
        pre.append(("norm", pio.make_normalizer("cp10k_log1p")))
    if selector != "none" and n_features > k:
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        # f_classif (ANOVA F) is the fast default — mutual_info's kNN density estimate is intractable
        # at ~35k RNA features inside nested CV. Both are SUPERVISED + fit per fold (leakage-safe).
        score = (partial(mutual_info_classif, random_state=0)
                 if selector == "mutual_info" else f_classif)
        pre.append(("fsel", SelectKBest(score, k=k)))
    out = {}
    for name, (est, grid) in models.build(list(config.models)).items():
        out[name] = (Pipeline(pre + list(est.steps)), grid)
    return out


# --------------------------------------------------------------------------- one (field, modality, cell-state)
def _skip(field, modality, cell_state, reason) -> dict:
    return {"field": field, "modality": modality, "cell_state": cell_state, "status": "skipped",
            "reason": reason, "weight": 0.0, "needs_more_data": True, "fidelity": _fidelity_tag(modality)}


def run_combo(ctx, field, modality, cell_state, config=None, labels=None, permutations=None) -> dict:
    """Learn one (field x modality x cell-state) model and harvest weight + OOF + validated markers."""
    config = config or DiscoveryConfig()
    base = {"field": field, "modality": modality, "cell_state": cell_state}
    if labels is None:
        labels = labels_for_field(ctx, field)
    labels = labels.dropna()
    usable = targets.usable_classes(labels, config.min_class_n)
    labels = labels[labels.isin(usable)]
    if labels.nunique() < 2:
        return _skip(field, modality, cell_state, "<2 usable classes")

    X, skv, donor, cohort = _combo_frame(ctx, modality, cell_state, set(labels.index))
    if X.shape[0] == 0:
        return _skip(field, modality, cell_state, "no pseudobulks/features")
    y = pd.Series([labels[k] for k in skv], index=X.index).astype(str)

    # gates (pseudobulk-level analogue of assemble_features' donor-group guards)
    n_pb, n_don = int(X.shape[0]), int(len(set(donor)))
    if n_pb < config.min_pseudobulks:
        return {**_skip(field, modality, cell_state, f"<{config.min_pseudobulks} pseudobulks"), "n_pseudobulks": n_pb}
    if n_don < config.min_donors:
        return {**_skip(field, modality, cell_state, f"<{config.min_donors} donor groups"), "n_pseudobulks": n_pb, "n_donors": n_don}
    dpc = pd.DataFrame({"y": y.values, "g": donor}).groupby("y")["g"].nunique()
    keep = set(dpc[dpc >= config.min_donors_per_class].index)
    if len(keep) < 2:
        return {**_skip(field, modality, cell_state, f"<2 classes with >={config.min_donors_per_class} donor groups"),
                "n_pseudobulks": n_pb, "n_donors": n_don}
    mask = y.isin(keep).values
    X, y = X.loc[mask], y[mask]
    donor, cohort = np.asarray(donor)[mask], np.asarray(cohort)[mask]

    # UNSUPERVISED top-variance pre-filter (label-independent -> leakage-safe): caps very wide modalities
    # (RNA ~35.7k, GRN ~7.5k) so the permutation baseline's full-feature reference estimator is tractable.
    if X.shape[1] > config.prefilter_features:
        keep_cols = X.var(axis=0).sort_values(ascending=False).index[:config.prefilter_features]
        X = X[list(keep_cols)]

    perms = permutations if permutations is not None else config.final_permutations
    # seam 6: per-combo feature-selection choice (default {} -> config defaults; identical behavior)
    try:
        sel = ctx.hooks.discovery_feature_select(field, modality, cell_state, int(X.shape[1])) or {}
    except Exception:
        sel = {}
    facto = _factories(modality, config, X.shape[1],
                       selector=sel.get("selector"), k=sel.get("k"))
    cvres = cv.nested_cv_evaluate(X, y, donor, cohort, strategy=config.strategy,
                                  model_factories=facto, outer_splits=5, inner_splits=3,
                                  n_permutations=perms)
    if cvres.get("error"):
        return {**base, "status": "error", "reason": cvres["error"], "weight": 0.0,
                "needs_more_data": True, "n_pseudobulks": n_pb, "n_donors": n_don,
                "fidelity": _fidelity_tag(modality)}

    ba = cvres.get("balanced_accuracy")
    p95 = cvres.get("permutation_balanced_accuracy_p95") or 0.0
    pval = cvres.get("permutation_pvalue")
    weight = 0.0
    if ba is not None and pval is not None and pval < config.alpha and p95 < 1.0:
        weight = float(np.clip((ba - p95) / (1.0 - p95), 0.0, 1.0))
    gate = ctx.hooks.gate_result(cvres)

    Xf = X
    if config.normalization.get(modality, "none") == "cp10k_log1p":
        Xf = pd.DataFrame(pio.cp10k_log1p(X.values), index=X.index, columns=X.columns)
    markers = cv.validated_markers(Xf, y, cvres.get("oof", {}), k=config.marker_k)
    wins = cvres.get("model_wins", {})
    return {**base, "status": "ok", "n_pseudobulks": int(X.shape[0]), "n_donors": int(len(set(donor))),
            "n_classes": int(y.nunique()), "balanced_accuracy": ba,
            "permutation_p": (round(float(pval), 4) if pval is not None else None),
            "permutation_p95": p95, "fold_std": cvres.get("balanced_accuracy_fold_std"),
            "weight": round(weight, 4), "gate_accept": bool(gate.accept), "gate_reason": gate.reason,
            "winning_model": (max(wins, key=wins.get) if wins else None),
            "fidelity": _fidelity_tag(modality), "needs_more_data": bool(weight == 0.0),
            "classes": cvres.get("classes"), "oof": cvres.get("oof", {}), "markers": markers,
            "feature_space": list(X.columns)}   # prefiltered input space -> the Deploy projection seam


# =========================================================================== Stage 3: full iteration
MEASURED = ("composition", "RNA", "cell-communication")   # imputed-from-RNA modalities are the complement
_HEAVY_KEYS = ("oof", "markers", "feature_space")          # large per-combo payloads (kept out of the weights table)


def _rank_fields(ctx, fields, config):
    """Feasibility-rank fields (fail-fast): runnable (>=2 usable classes) first, then by #labeled samples.
    Returns (plan=[(field, n_classes, n_labeled)], labels_map={field: usable-label Series})."""
    plan, labels_map = [], {}
    for f in fields:
        try:
            lab = labels_for_field(ctx, f).dropna()
        except Exception:
            lab = pd.Series(dtype=object)
        lab = lab[lab.isin(targets.usable_classes(lab, config.min_class_n))]
        labels_map[f] = lab
        plan.append((f, int(lab.nunique()), int(lab.shape[0])))
    plan.sort(key=lambda t: (t[1] >= 2, t[2], t[1]), reverse=True)
    return plan, labels_map


def _log_combo(r) -> None:
    f, m, c = r.get("field"), r.get("modality"), r.get("cell_state")
    st = r.get("status")
    if st == "ok":
        print(f"  [ok      ] {f} x {m} x {c}: ba={r.get('balanced_accuracy')} "
              f"p={r.get('permutation_p')} w={r.get('weight')} ({r.get('pass')})", flush=True)
    elif st == "skipped":
        print(f"  [skip    ] {f} x {m} x {c}: {r.get('reason')}", flush=True)
    else:
        print(f"  [{st:8s}] {f} x {m} x {c}: {r.get('reason')}", flush=True)


def _screen_confirm(ctx, field, modality, cell_state, config, labels, log=True):
    """Two-pass permutation tiering: screen cheaply, then re-run only survivors with the precise
    permutation count. Single pass when final<=screen or final==0 (e.g. smoke tests). The CV/OOF/markers
    are deterministic (seed=0) so the confirm pass reproduces them and only sharpens the p-value/weight;
    the (small) re-cost is dwarfed by the precise permutation baseline it is paired with."""
    single = (config.final_permutations == 0) or (config.final_permutations <= config.screen_permutations)
    r = run_combo(ctx, field, modality, cell_state, config, labels=labels,
                  permutations=config.screen_permutations)
    r["pass"] = "single" if single else "screen"
    if (not single) and r.get("status") == "ok":
        sp = r.get("permutation_p")
        if sp is not None and sp <= config.screen_promote_alpha:
            r = run_combo(ctx, field, modality, cell_state, config, labels=labels,
                          permutations=config.final_permutations)
            r["pass"] = "confirm"
    if log:
        _log_combo(r)
    return r


def run_discovery(ctx, config=None, fields=None, modalities=None, cell_states_top=40,
                  sample_level=True, write=True, verbose=True) -> dict:
    """Iterate (field x modality x cell-state) -> per-combo weight/OOF/marker harvest, with skip-gates,
    fail-fast field ordering, population-ranked cell-states, and two-pass permutation tiering. Writes the
    three Discovery tables (+ association index) and the report. The baseline path is untouched.

    fields: subset of `candidate_fields` (default all). modalities: subset (default all pseudobulk +,
    if sample_level, the sample-level set). cell_states_top: top-N cell-states by population (pseudobulk
    modalities only). Returns {results, weights, markers, associations, report}."""
    config = config or DiscoveryConfig()
    if "mutations" not in ctx.tables:
        try:
            ctx.tables["mutations"] = genetics.build_mutation_matrix(ctx)
        except Exception:
            pass

    req = list(modalities) if modalities else (
        list(PSEUDOBULK_MODALITIES) + (list(SAMPLE_LEVEL) if sample_level else []))
    pb_mods = [m for m in req if m in PSEUDOBULK_MODALITIES and m in ctx._modality_paths]
    sl_mods = [m for m in req if m in SAMPLE_LEVEL]

    sizes = pio.cell_state_sizes(ctx)
    states = [cs for cs, n in sizes.items() if int(n) >= config.min_pseudobulks][:int(cell_states_top)]

    plan, labels_map = _rank_fields(ctx, fields or candidate_fields(ctx), config)
    if verbose:
        print(f"[discovery] {len(plan)} fields | pseudobulk modalities {pb_mods} x {len(states)} states "
              f"| sample-level {sl_mods} | run_id={ctx.config.run_id}", flush=True)

    results = []
    with warnings.catch_warnings():
        # benign + very chatty under multiclass small folds: a fold's predictions can include a class
        # absent from that fold's held-out truth. Does not affect the OOF metric. Quiet it for the sweep.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        for field, ncls, nlab in plan:
            if ncls < 2:
                results.append(_skip(field, "*", "*", f"<2 usable classes (field-level; {nlab} labeled)"))
                if verbose:
                    _log_combo(results[-1])
                continue
            if verbose:
                print(f"[discovery] field '{field}' ({ncls} classes, {nlab} labeled)", flush=True)
            labels = labels_map[field]
            for mod in sl_mods:
                results.append(_screen_confirm(ctx, field, mod, SAMPLE_SENTINEL, config, labels, verbose))
            for mod in pb_mods:
                for cs in states:
                    results.append(_screen_confirm(ctx, field, mod, cs, config, labels, verbose))

    weights = _weights_table(results)
    markers = _markers_table(ctx, results)
    assoc, assoc_index = _associations(results, config)
    report = _build_report(ctx, results, config, labels_map)

    if write:
        ctx.save_table(weights, "discovery_weights.tsv", index=False)
        ctx.save_table(markers, "discovery_markers.tsv", index=False)
        ctx.save_table(assoc, "discovery_associations.tsv", index=False)
        ctx.save_json(assoc_index, "discovery_associations_index.json")
        ctx.save_json(report, "discovery_report.json")
        with open(ctx.path("DISCOVERY.md"), "w", encoding="utf-8") as fh:
            fh.write(_render_markdown(report))
        if verbose:
            print(f"[discovery] wrote 3 tables + index + report to {ctx.run_dir}", flush=True)
    return {"results": results, "weights": weights, "markers": markers,
            "associations": assoc, "association_index": assoc_index, "report": report}


# --------------------------------------------------------------------------- output tables
def _weights_table(results) -> pd.DataFrame:
    cols = ["field", "modality", "cell_state", "status", "pass", "reason",
            "n_pseudobulks", "n_donors", "n_classes", "balanced_accuracy", "permutation_p",
            "permutation_p95", "weight", "gate_accept", "winning_model", "fidelity", "needs_more_data",
            "classes"]
    rows = []
    for r in results:
        row = {k: r.get(k) for k in cols}
        cl = r.get("classes")
        row["classes"] = "|".join(map(str, cl)) if isinstance(cl, (list, tuple)) else cl
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    # most-actionable first: significant by weight desc, then the rest
    return df.sort_values(["weight", "balanced_accuracy"], ascending=False, na_position="last",
                          kind="mergesort").reset_index(drop=True)


def _markers_table(ctx, results) -> pd.DataFrame:
    """Held-out-VALIDATED discriminating features only (train AUROC + held-out OOF AUROC + provenance).
    For imputed modalities, attaches the feature's held-out Spearman fidelity when the metric exists."""
    cols = ["field", "modality", "cell_state", "feature", "rf_importance", "train_auroc",
            "heldout_auroc", "combo_weight", "fidelity", "feature_heldout_spearman"]
    fid_cache, rows = {}, []
    for r in results:
        if r.get("status") != "ok":
            continue
        mod = r["modality"]
        for m in r.get("markers", []):
            if not m.get("heldout_validated"):
                continue
            fhs = None
            if mod in IMPUTED_MODALITIES:
                if mod not in fid_cache:
                    try:
                        fid_cache[mod] = dataio.feature_fidelity(ctx, mod)
                    except Exception:
                        fid_cache[mod] = None
                fid = fid_cache[mod]
                if fid is not None and m["feature"] in getattr(fid, "index", []):
                    fhs = round(float(fid[m["feature"]]), 3)
            rows.append({"field": r["field"], "modality": mod, "cell_state": r["cell_state"],
                         "feature": m["feature"], "rf_importance": m.get("rf_importance"),
                         "train_auroc": m.get("train_auroc"), "heldout_auroc": m.get("heldout_auroc"),
                         "combo_weight": r.get("weight"), "fidelity": r.get("fidelity"),
                         "feature_heldout_spearman": fhs})
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["field", "modality", "combo_weight", "rf_importance"],
                          ascending=[True, True, False, False], kind="mergesort").reset_index(drop=True)


def _associations(results, config):
    """Per-pseudobulk held-out OOF {true,pred,prob} per combo (the Deploy NN seam) + an index JSON
    carrying each combo's feature space (for combos worth deploying) + metadata for `load_discovery`."""
    cols = ["field", "modality", "cell_state", "pseudobulk_id", "true", "pred", "prob"]
    rows, index = [], {}
    for r in results:
        if r.get("status") != "ok":
            continue
        key = "%s|%s|%s" % (r["field"], r["modality"], r["cell_state"])
        for pid, d in r.get("oof", {}).items():
            rows.append({"field": r["field"], "modality": r["modality"], "cell_state": r["cell_state"],
                         "pseudobulk_id": pid, "true": d.get("true"), "pred": d.get("pred"),
                         "prob": d.get("prob")})
        entry = {"field": r["field"], "modality": r["modality"], "cell_state": r["cell_state"],
                 "weight": r.get("weight"), "permutation_p": r.get("permutation_p"),
                 "balanced_accuracy": r.get("balanced_accuracy"), "n_pseudobulks": r.get("n_pseudobulks"),
                 "classes": r.get("classes"), "winning_model": r.get("winning_model"),
                 "fidelity": r.get("fidelity"),
                 "normalization": config.normalization.get(r["modality"], "none")}
        if (r.get("weight") or 0) > 0 or r.get("gate_accept"):
            entry["feature_space"] = r.get("feature_space", [])   # only stored where Deploy would query it
        index[key] = entry
    return pd.DataFrame(rows, columns=cols), index


# --------------------------------------------------------------------------- report
def _config_dict(config) -> dict:
    d = asdict(config)
    d["models"] = list(d.get("models", ()))
    return d


def _cohort_confound(ctx, labels, config, frac=0.8):
    """Flag classes that come overwhelmingly from one cohort/dataset (donor_kfold can't separate cohort
    signal from biology) -> recommend leave_one_cohort_out. Returns a string or None."""
    if labels is None or labels.shape[0] == 0 or "samples" not in ctx.tables:
        return None
    ds = ctx.tables["samples"].reindex(labels.index)["dataset"].astype(str)
    df = pd.DataFrame({"y": labels.values, "ds": ds.values}).dropna()
    flags = []
    for cls, g in df.groupby("y"):
        if g.shape[0] < config.min_class_n:
            continue
        vc = g["ds"].value_counts(normalize=True)
        if len(vc) and float(vc.iloc[0]) >= frac:
            flags.append("%s~%s(%.0f%%)" % (cls, vc.index[0], 100 * float(vc.iloc[0])))
    return "; ".join(flags) if flags else None


def _optimize_me(ctx, results, field_ability, labels_map, config) -> list:
    """Rule-derived 'poorly-predicted / optimize-me' list: imputed-only signal (circularity caution),
    donor/class-floor skips (needs more data), ran-but-nothing (field may not be encoded), and
    cohort-confounded predictable fields (recommend leave_one_cohort_out)."""
    by_field = {}
    for r in results:
        by_field.setdefault(r["field"], []).append(r)
    out = []
    for f, fa in field_ability.items():
        rs = by_field.get(f, [])
        ran = [r for r in rs if r.get("status") == "ok"]
        skipped = [r for r in rs if r.get("status") == "skipped"]
        if fa["imputed_weight"] > 0 and fa["measured_weight"] == 0:
            out.append({"field": f, "issue": "signal only in imputed-from-RNA modalities",
                        "recommendation": "confirm on RNA/composition before trusting (possible circularity)"})
        if not ran and skipped:
            reasons = sorted({r.get("reason") for r in skipped if r.get("reason")})
            out.append({"field": f, "issue": "all combos skipped (%s)" % ("; ".join(reasons)[:140]),
                        "recommendation": "needs more donors/samples or more usable classes"})
        elif ran and fa["best_weight"] == 0:
            out.append({"field": f, "issue": "ran but no modality predicts above chance",
                        "recommendation": "needs_more_data: more samples, or not encoded in these modalities"})
        if fa["predictable"]:
            conf = _cohort_confound(ctx, labels_map.get(f), config)
            if conf:
                out.append({"field": f, "issue": "class cohort-confounded: %s" % conf,
                            "recommendation": "validate with leave_one_cohort_out (donor_kfold weight may be cohort-inflated)"})
    return out


def _build_report(ctx, results, config, labels_map) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    fields = sorted({r["field"] for r in results})
    field_ability, mutation_pred = {}, {}
    for f in fields:
        fr = [r for r in ok if r["field"] == f]
        best = max(fr, key=lambda r: (r.get("weight") or 0.0), default=None)
        sig = [r for r in fr if (r.get("weight") or 0.0) > 0]
        meas_w = max([(r.get("weight") or 0.0) for r in fr if r["modality"] in MEASURED], default=0.0)
        imp_w = max([(r.get("weight") or 0.0) for r in fr if r["modality"] not in MEASURED], default=0.0)
        field_ability[f] = {
            "predictable": bool(best and (best.get("weight") or 0.0) > 0),
            "best_weight": round(best.get("weight"), 4) if best else 0.0,
            "best_modality": best["modality"] if best else None,
            "best_cell_state": best["cell_state"] if best else None,
            "best_balanced_accuracy": best.get("balanced_accuracy") if best else None,
            "best_permutation_p": best.get("permutation_p") if best else None,
            "n_combos_run": len(fr), "n_combos_significant": len(sig),
            "measured_weight": round(meas_w, 4), "imputed_weight": round(imp_w, 4)}
        if f.startswith(("mut_", "cyto_")):
            per_mod = {}
            for r in fr:
                m, w = r["modality"], (r.get("weight") or 0.0)
                if m not in per_mod or w > per_mod[m]["weight"]:
                    per_mod[m] = {"weight": round(w, 4), "cell_state": r["cell_state"],
                                  "permutation_p": r.get("permutation_p"),
                                  "balanced_accuracy": r.get("balanced_accuracy")}
            mutation_pred[f] = per_mod

    optimize = _optimize_me(ctx, results, field_ability, labels_map, config)
    summary = {
        "n_combos": len(results), "n_ok": len(ok),
        "n_significant": sum(1 for r in ok if (r.get("weight") or 0.0) > 0),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "fields_predictable": sorted(f for f, a in field_ability.items() if a["predictable"]),
        "fields_needs_more_data": sorted(f for f, a in field_ability.items() if not a["predictable"])}
    return {"run_id": ctx.config.run_id, "config": _config_dict(config), "summary": summary,
            "field_ability": field_ability, "mutation_predictability": mutation_pred,
            "optimize_me": optimize}


def _render_markdown(report) -> str:
    s = report.get("summary", {})
    L = ["# Discovery report: %s" % report.get("run_id", "?"), "",
         "Per-(metadata field x modality x cell-state) **permutation-calibrated weights** from "
         "donor-grouped 5-fold CV at pseudobulk resolution. Weight 0 = no signal above chance (honest, "
         "not faked). Imputed-from-RNA modalities are flagged — signal there but not in RNA/composition "
         "is treated as possible circularity.", "",
         "- combos: **%s** (ok %s, skipped %s, error %s)  ·  significant: **%s**"
         % (s.get("n_combos"), s.get("n_ok"), s.get("n_skipped"), s.get("n_error"), s.get("n_significant")),
         "- predictable fields: %s" % (", ".join(s.get("fields_predictable", [])) or "_none_"),
         "- needs-more-data fields: %s" % (", ".join(s.get("fields_needs_more_data", [])) or "_none_"), ""]

    L += ["## Field predictability", "",
          "| field | predictable | best weight | best modality | best cell-state | best bal.acc | perm p | measured w | imputed w | combos sig/run |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    fa = report.get("field_ability", {})
    for f in sorted(fa, key=lambda k: fa[k]["best_weight"], reverse=True):
        a = fa[f]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s/%s |" % (
            f, "✓" if a["predictable"] else "—", a["best_weight"], a["best_modality"] or "—",
            a["best_cell_state"] or "—", a["best_balanced_accuracy"], a["best_permutation_p"],
            a["measured_weight"], a["imputed_weight"], a["n_combos_significant"], a["n_combos_run"]))

    mp = report.get("mutation_predictability", {})
    if mp:
        mods = sorted({m for d in mp.values() for m in d})
        L += ["", "## Mutation-predictability matrix (best weight per modality, across cell-states)", "",
              "| mutation/cyto | " + " | ".join(mods) + " |",
              "|" + "---|" * (len(mods) + 1)]
        for f in sorted(mp, key=lambda k: max([v["weight"] for v in mp[k].values()] or [0]), reverse=True):
            cells = []
            for m in mods:
                v = mp[f].get(m)
                cells.append(("%s" % v["weight"]) if v and v["weight"] > 0 else "·")
            L.append("| %s | %s |" % (f, " | ".join(cells)))

    opt = report.get("optimize_me", [])
    if opt:
        L += ["", "## Optimize-me (rule-derived)", ""]
        for o in opt:
            L.append("- **%s** — %s → _%s_" % (o["field"], o["issue"], o["recommendation"]))
    L.append("")
    return "\n".join(L)


# =========================================================================== Stage 5: typed loader (Deploy/Arbiter seam)
@dataclass
class DiscoveryResult:
    """Typed view over a Discovery run dir — what Deploy and the Arbiter consume. `associations` is the
    NN seam (per-pseudobulk OOF {true,pred,prob}); `association_index` carries per-combo metadata +
    `feature_space` (where stored) for projecting a new specimen; `weight()` moderates Deploy/Arbiter
    votes; `report` holds field_ability + the mutation-predictability matrix."""
    run_dir: str
    weights: pd.DataFrame
    markers: pd.DataFrame
    associations: pd.DataFrame
    association_index: dict
    report: dict

    @staticmethod
    def _key(field, modality, cell_state) -> str:
        return "%s|%s|%s" % (field, modality, cell_state)

    def combo(self, field, modality, cell_state) -> "dict | None":
        return self.association_index.get(self._key(field, modality, cell_state))

    def weight(self, field, modality, cell_state) -> float:
        e = self.combo(field, modality, cell_state)
        w = e.get("weight") if e else None
        return float(w) if w is not None else 0.0

    def feature_space(self, field, modality, cell_state) -> list:
        e = self.combo(field, modality, cell_state)
        return list(e.get("feature_space", [])) if e else []

    def associations_for(self, field, modality, cell_state) -> pd.DataFrame:
        a = self.associations
        if a.empty:
            return a
        return a[(a["field"] == field) & (a["modality"] == modality) & (a["cell_state"] == cell_state)]

    def field_ability(self) -> dict:
        return self.report.get("field_ability", {})

    def mutation_predictability(self) -> dict:
        return self.report.get("mutation_predictability", {})


def load_discovery(run_dir) -> DiscoveryResult:
    """Load a Discovery run dir into a typed DiscoveryResult. Missing/empty outputs load as empty frames
    or {} (forward-compatible: extra columns / unknown status values are preserved, not rejected)."""
    def _tsv(name):
        p = os.path.join(run_dir, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                return pd.read_csv(p, sep="\t")
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _json(name):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    return DiscoveryResult(
        run_dir=str(run_dir),
        weights=_tsv("discovery_weights.tsv"),
        markers=_tsv("discovery_markers.tsv"),
        associations=_tsv("discovery_associations.tsv"),
        association_index=_json("discovery_associations_index.json"),
        report=_json("discovery_report.json"))


def merge_discovery(run_dirs, out_dir) -> dict:
    """Merge per-field Discovery run dirs (the LSF fan-out outputs) into one combined sweep dir.
    Concatenates the three tables, unions the association index, and unions the per-field reports
    (field entries are disjoint across per-field jobs) with a recomputed summary + re-rendered
    DISCOVERY.md. NO CV is re-run — this only stitches finished outputs. Returns the combined report."""
    os.makedirs(out_dir, exist_ok=True)
    drs = [(d, load_discovery(d)) for d in run_dirs]
    drs = [(d, r) for d, r in drs if len(r.weights) or r.report]

    def _cat(attr):
        frames = [getattr(r, attr) for _, r in drs if len(getattr(r, attr))]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    weights, markers, assoc = _cat("weights"), _cat("markers"), _cat("associations")
    if len(weights):
        weights = weights.drop_duplicates(["field", "modality", "cell_state"], keep="last")

    index, field_ability, mutation_pred = {}, {}, {}
    optimize = []
    for _, r in drs:
        index.update(r.association_index)
        field_ability.update(r.report.get("field_ability", {}))
        mutation_pred.update(r.report.get("mutation_predictability", {}))
        optimize.extend(r.report.get("optimize_me", []))

    okw = weights[weights["status"] == "ok"] if len(weights) else weights
    summary = {
        "n_combos": int(len(weights)), "n_ok": int(len(okw)),
        "n_significant": int((okw["weight"] > 0).sum()) if len(okw) else 0,
        "n_skipped": int((weights["status"] == "skipped").sum()) if len(weights) else 0,
        "n_error": int((weights["status"] == "error").sum()) if len(weights) else 0,
        "fields_predictable": sorted(f for f, a in field_ability.items() if a.get("predictable")),
        "fields_needs_more_data": sorted(f for f, a in field_ability.items() if not a.get("predictable"))}
    report = {"run_id": os.path.basename(out_dir.rstrip("/\\")),
              "merged_from": [os.path.basename(d.rstrip("/\\")) for d, _ in drs],
              "summary": summary, "field_ability": field_ability,
              "mutation_predictability": mutation_pred, "optimize_me": optimize}

    weights.to_csv(os.path.join(out_dir, "discovery_weights.tsv"), sep="\t", index=False)
    markers.to_csv(os.path.join(out_dir, "discovery_markers.tsv"), sep="\t", index=False)
    assoc.to_csv(os.path.join(out_dir, "discovery_associations.tsv"), sep="\t", index=False)
    with open(os.path.join(out_dir, "discovery_associations_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, default=str)
    with open(os.path.join(out_dir, "discovery_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, "DISCOVERY.md"), "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))
    return report
