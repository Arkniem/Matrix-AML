"""Leakage-proof cross-validation harness — the correctness core.

Honest, group-aware nested CV. The whole project's credibility rests here, so the
rules are explicit and enforced:

  * Outer CV produces the reported (out-of-fold) metrics. Strategy is either
    donor-grouped StratifiedGroupKFold or leave-one-cohort-out.
  * Inner CV (always donor-grouped) selects the model/hyperparameters INSIDE each
    outer-train fold. No information from the outer-test fold ever touches model
    selection or preprocessing (preprocessing lives in the Pipeline -> fit per fold).
  * A hard assertion verifies no donor appears in both train and test of any fold.
  * A within-data permutation baseline gives the chance ceiling (p95) and an
    empirical p-value, so "above chance" is quantified, not assumed.

This is exactly the discipline that turned a reported 0.85 into an honest 0.59 in
the lab's LSC bake-off. Do not let an agent replace this with its own splitting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupKFold, LeaveOneGroupOut, GridSearchCV)
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from .models import simple_estimator, NJOBS


def assert_no_leakage(train_groups, test_groups) -> None:
    overlap = set(map(str, train_groups)) & set(map(str, test_groups))
    if overlap:
        raise AssertionError(
            f"GROUP LEAKAGE: {len(overlap)} group(s) in both train and test "
            f"(e.g. {list(overlap)[:3]}). Grouped CV is broken.")


def _max_safe_splits(requested, y, groups) -> int:
    """Cap folds at the smallest class's number of distinct groups (>=2)."""
    df = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)})
    per_class = df.groupby("y")["g"].nunique()
    return max(2, min(int(requested), int(per_class.min())))


def _outer_iter(strategy, X, y, groups_donor, groups_cohort, n_splits, seed):
    if strategy == "leave_one_cohort_out":
        if groups_cohort is None:
            raise ValueError("leave_one_cohort_out requires groups_cohort")
        splitter = LeaveOneGroupOut()
        for tr, te in splitter.split(X, y, groups=np.asarray(groups_cohort)):
            yield tr, te
    elif strategy == "donor_kfold":
        k = _max_safe_splits(n_splits, y, groups_donor)
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr, te in splitter.split(X, y, groups=np.asarray(groups_donor)):
            yield tr, te
    else:
        raise ValueError(f"unknown strategy '{strategy}'")


def _inner_cv(y_train, groups_train, inner_splits):
    # stratified + grouped so inner model-selection folds keep every class present
    k = _max_safe_splits(inner_splits, y_train, groups_train)
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)


def nested_cv_evaluate(X, y, groups_donor, groups_cohort=None,
                       strategy="donor_kfold", model_factories=None,
                       outer_splits=5, inner_splits=3, seed=0,
                       n_permutations=100):
    """Run honest nested grouped CV. Returns a json-serializable result dict.

    X: 2D array/DataFrame (n_samples x n_features). y: 1D labels.
    groups_donor: donor id per sample (always used for inner CV + leakage check).
    groups_cohort: dataset/cohort per sample (used when strategy=leave_one_cohort_out).
    model_factories: {name: (estimator_pipeline, param_grid)} from amlmm.models.build.
    """
    index = list(X.index) if hasattr(X, "index") else list(range(len(y)))
    X = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
    y = np.asarray(y).astype(str)
    groups_donor = np.asarray(groups_donor).astype(str)
    if groups_cohort is not None:
        groups_cohort = np.asarray(groups_cohort).astype(str)
    if model_factories is None:
        from .models import build
        model_factories = build(["rf", "logreg"])

    classes = sorted(pd.unique(y).tolist())
    if strategy == "leave_one_cohort_out":
        if groups_cohort is None or len(set(map(str, groups_cohort))) < 2:
            return {"error": "leave_one_cohort_out requires >=2 cohorts",
                    "n_samples": int(len(y)), "classes": classes, "strategy": strategy}
    oof_true, oof_pred = [], []
    oof = {}              # sample -> {true, pred, prob}; held-out per-sample predictions
    per_fold, winners = [], []

    for fold, (tr, te) in enumerate(
            _outer_iter(strategy, X, y, groups_donor, groups_cohort, outer_splits, seed)):
        assert_no_leakage(groups_donor[tr], groups_donor[te])
        inner = _inner_cv(y[tr], groups_donor[tr], inner_splits)
        best_name, best_est, best_score = None, None, -np.inf
        for name, (est, grid) in model_factories.items():
            try:
                gs = GridSearchCV(est, grid, cv=inner, scoring="balanced_accuracy",
                                  n_jobs=NJOBS, refit=True, error_score=np.nan)
                gs.fit(X[tr], y[tr], groups=groups_donor[tr])
            except Exception:
                continue
            if np.isfinite(gs.best_score_) and gs.best_score_ > best_score:
                best_name, best_est, best_score = name, gs.best_estimator_, gs.best_score_
        if best_est is None:
            continue
        pred = best_est.predict(X[te])
        proba = best_est.predict_proba(X[te]) if hasattr(best_est, "predict_proba") else None
        for j, ti in enumerate(te):
            oof[str(index[ti])] = {"true": str(y[ti]), "pred": str(pred[j]),
                                   "prob": (round(float(np.max(proba[j])), 4) if proba is not None else None)}
        oof_true.extend(y[te].tolist())
        oof_pred.extend(np.asarray(pred).astype(str).tolist())
        per_fold.append({
            "fold": fold, "n_test": int(len(te)),
            "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
            "model": best_name, "inner_score": round(float(best_score), 4),
            "test_cohort": (sorted(set(groups_cohort[te]))[0]
                            if strategy == "leave_one_cohort_out" else None),
        })
        winners.append(best_name)

    if not oof_true:
        return {"error": "no folds produced predictions", "n_samples": int(len(y)),
                "classes": classes, "strategy": strategy}

    oof_true = np.array(oof_true); oof_pred = np.array(oof_pred)
    ba = float(balanced_accuracy_score(oof_true, oof_pred))
    macro_f1 = float(f1_score(oof_true, oof_pred, average="macro", labels=classes, zero_division=0))
    per_class_f1 = {c: float(f) for c, f in zip(
        classes, f1_score(oof_true, oof_pred, average=None, labels=classes, zero_division=0))}
    cm = confusion_matrix(oof_true, oof_pred, labels=classes).tolist()

    perm = _permutation_baseline(X, y, groups_donor, groups_cohort, strategy,
                                 outer_splits, seed, n_permutations)

    fold_scores = [f["balanced_accuracy"] for f in per_fold]
    return {
        "strategy": strategy,
        "n_samples": int(len(y)),
        "n_classes": len(classes),
        "classes": classes,
        "class_counts": {c: int((y == c).sum()) for c in classes},
        "balanced_accuracy": round(ba, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        "balanced_accuracy_fold_mean": round(float(np.mean(fold_scores)), 4),
        "balanced_accuracy_fold_std": round(float(np.std(fold_scores)), 4),
        "confusion_matrix": cm,
        "model_wins": {m: int(winners.count(m)) for m in model_factories},
        "per_fold": per_fold,
        "oof": oof,
        "permutation_balanced_accuracy_p95": perm["p95"],
        "permutation_balanced_accuracy_mean": perm["mean"],
        "permutation_observed_reference": perm.get("observed_reference"),
        "permutation_pvalue": perm["pvalue"],
        "n_permutations": perm["n"],
    }


def _permutation_baseline(X, y, groups_donor, groups_cohort, strategy,
                          outer_splits, seed, n_permutations):
    """Honest chance ceiling with a fixed reference estimator.

    Two corrections vs a naive permutation test on grouped data:
      * FIXED, label-independent folds (GroupKFold on donors / LeaveOneGroupOut on
        cohorts) computed ONCE and reused for the observed score and every
        permutation -- only the labels change between iterations, never the partition.
      * GROUP-LEVEL label shuffle -- the label is a donor-level constant in the real
        task, so we permute one label per donor group and broadcast, preserving
        within-donor label constancy (and class frequencies). A per-sample shuffle
        would scatter labels within a donor and bias the null downward.
    `observed_reference` is the same reference estimator on TRUE labels and fixed
    folds, so `pvalue` is a self-consistent significance test (used by the gate)."""
    rng = np.random.default_rng(seed)
    groups_donor = np.asarray(groups_donor).astype(str)

    if strategy == "leave_one_cohort_out":
        folds = list(LeaveOneGroupOut().split(X, y, groups=np.asarray(groups_cohort).astype(str)))
    else:
        k = _max_safe_splits(outer_splits, y, groups_donor)
        folds = list(GroupKFold(n_splits=k).split(X, y, groups=groups_donor))

    def oof_ba(yy):
        t, p = [], []
        for tr, te in folds:
            est = simple_estimator()
            try:
                est.fit(X[tr], yy[tr]); pr = est.predict(X[te])
            except Exception:
                continue
            t.extend(yy[te].tolist()); p.extend(np.asarray(pr).astype(str).tolist())
        return balanced_accuracy_score(np.array(t), np.array(p)) if t else np.nan

    uniq = pd.unique(groups_donor)
    group_label = np.array([pd.Series(y[groups_donor == g]).value_counts().index[0] for g in uniq])

    def shuffled_y():
        perm = rng.permutation(len(uniq))
        m = {uniq[i]: group_label[perm[i]] for i in range(len(uniq))}
        return np.array([m[g] for g in groups_donor])

    observed = oof_ba(y)
    null = np.array([v for v in (oof_ba(shuffled_y()) for _ in range(int(n_permutations)))
                     if np.isfinite(v)])
    if null.size == 0:
        return {"p95": None, "mean": None, "pvalue": None, "n": 0, "observed_reference": None}
    pval = float((np.sum(null >= observed) + 1) / (null.size + 1)) if np.isfinite(observed) else None
    return {"p95": round(float(np.percentile(null, 95)), 4),
            "mean": round(float(null.mean()), 4),
            "pvalue": pval, "n": int(null.size),
            "observed_reference": round(float(observed), 4) if np.isfinite(observed) else None}


def _ovr_auroc(values, labels) -> float:
    """Macro one-vs-rest AUROC of a single feature's values vs multiclass labels. Rank-based
    (direction-agnostic: max(a, 1-a)), so invariant to monotone normalization; nan if undefined."""
    from sklearn.metrics import roc_auc_score
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels).astype(str)
    if len(v) < 4 or np.unique(lab).size < 2 or np.all(v == v[0]):
        return float("nan")
    aucs = []
    for c in np.unique(lab):
        yb = (lab == c).astype(int)
        if 0 < yb.sum() < len(yb):
            try:
                a = roc_auc_score(yb, v)
                aucs.append(max(a, 1.0 - a))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def validated_markers(Xf, y, oof, k=15, min_heldout_auroc=0.6) -> list:
    """Top features by a full-data RF importance, RETAINED only if they ALSO separate classes on the
    HELD-OUT (out-of-fold) rows — "validated on held-out". Xf: an appropriately-normalized DataFrame
    [rows x features] aligned to y (a Series). oof: {row_id: {true,pred,prob}} from a cv_result.
    Returns [{feature, rf_importance, train_auroc, heldout_auroc, heldout_validated}] (importance desc)."""
    if Xf is None or getattr(Xf, "shape", (0, 0))[0] < 6 or Xf.shape[1] == 0:
        return []
    from sklearn.ensemble import RandomForestClassifier
    yv = y.reindex(Xf.index).astype(str).values
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                random_state=0, n_jobs=NJOBS)
    try:
        rf.fit(np.asarray(Xf.values, dtype=float), yv)
    except Exception:
        return []
    imp = rf.feature_importances_
    cols = list(Xf.columns)
    oof_ids = [i for i in Xf.index if str(i) in oof]
    Xoof = Xf.loc[oof_ids] if oof_ids else None
    yoof = np.array([oof[str(i)]["true"] for i in oof_ids]) if oof_ids else None
    out = []
    for j in np.argsort(imp)[::-1][:int(k)]:
        feat = cols[j]
        tr = _ovr_auroc(Xf.iloc[:, j].values, yv)
        ho = (_ovr_auroc(Xoof[feat].values, yoof) if (Xoof is not None and len(oof_ids) >= 4)
              else float("nan"))
        out.append({"feature": feat, "rf_importance": round(float(imp[j]), 5),
                    "train_auroc": (None if np.isnan(tr) else round(tr, 3)),
                    "heldout_auroc": (None if np.isnan(ho) else round(ho, 3)),
                    "heldout_validated": bool(not np.isnan(ho) and ho >= min_heldout_auroc)})
    return out
