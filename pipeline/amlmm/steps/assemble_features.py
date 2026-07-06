"""Step: assemble_features — build (X, y, groups) for one target.

Label = the canonical target (subtype etc.), restricted to usable classes.
Feature blocks are chosen via the DecisionHooks seam (default: composition only).
Each block is a sample x feature matrix:
  * composition  — cell-state fractions (cell frequency); fully measured.
  * <modality>   — n_cells-weighted mean of the sample's pseudobulks; imputed
                   modalities are filtered to features with held-out Spearman >= 0.3.
Blocks are concatenated (column prefix = block name) and aligned to the samples
present in every chosen block. Sets ctx artifacts X, y, groups_donor, groups_cohort.
"""
from __future__ import annotations
import numpy as np

from ..step import StepSpec, StepResult, register
from .. import targets as T
from .. import dataio


def run(ctx, params) -> StepResult:
    target = params["target"]
    min_class_n = int(params["min_class_n"])
    lab = T.get_labels(ctx, target).dropna()
    usable = T.usable_classes(lab, min_class_n)
    lab = lab[lab.isin(usable)]
    if lab.nunique() < 2:
        return StepResult(name="assemble_features", status="skipped",
                          metrics={"reason": "<2 usable classes", "target": target,
                                   "usable_classes": usable})
    keys = list(lab.index)
    feas = (ctx.getart("feasibility", {}) or {}).get(target, {})
    available = ["composition"] + ctx.modalities()
    blocks = params.get("blocks") or ctx.hooks.select_feature_blocks(target, available, feas)

    mats = []
    for b in blocks:
        if b == "composition":
            comp = ctx.tables["composition"].reindex(keys)
            comp = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            comp.columns = [f"comp::{c}" for c in comp.columns]
            mats.append(comp)
        elif b in ctx.modalities():
            ms = 0.3 if b in dataio.IMPUTED_MODALITIES else None
            mm = dataio.sample_modality_matrix(ctx, b, sample_keys=keys, min_spearman=ms)
            mm.columns = [f"{b}::{c}" for c in mm.columns]
            mats.append(mm)
        else:
            return StepResult(name="assemble_features", status="error",
                              metrics={"reason": f"unknown block '{b}'"})

    common = set(keys)
    for m in mats:
        common &= set(m.index)
    common = [k for k in keys if k in common]
    if len(common) < 2 * lab.nunique():
        return StepResult(name="assemble_features", status="skipped",
                          metrics={"reason": "too few samples with all blocks",
                                   "n_common": len(common), "blocks": blocks})

    import pandas as pd
    s = ctx.tables["samples"]
    y = lab.reindex(common)
    dg = s.loc[common, "donor_group"]
    # A class needs >=2 distinct donor groups; with only 1, grouped CV forces it
    # entirely into a test fold (absent from training) and silently biases balanced acc.
    groups_per_class = pd.DataFrame({"y": y.values, "g": dg.values}).groupby("y")["g"].nunique()
    keep = set(groups_per_class[groups_per_class >= 2].index)
    dropped = sorted(set(y.unique()) - keep)
    common = [k for k in common if y.loc[k] in keep]
    y = lab.reindex(common)
    if y.nunique() < 2:
        return StepResult(name="assemble_features", status="skipped",
                          metrics={"reason": "<2 classes with >=2 donor groups (grouped-CV unlearnable)",
                                   "target": target, "dropped_single_group_classes": dropped})

    X = pd.concat([m.reindex(common) for m in mats], axis=1).fillna(0.0)
    ctx.set("X", X)
    ctx.set("y", y)
    ctx.set("groups_donor", s.loc[common, "donor_group"])
    ctx.set("groups_cohort", s.loc[common, "dataset"])
    ctx.set("target", target)
    ctx.set("feature_blocks", blocks)

    res = StepResult(name="assemble_features")
    if dropped:
        res.add_log(f"dropped single-donor-group classes (unlearnable under grouped CV): {dropped}")
    res.metrics = {
        "target": target, "blocks": blocks,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "dropped_single_group_classes": dropped,
        "n_donor_groups": int(s.loc[common, "donor_group"].nunique()),
        "n_cohorts": int(s.loc[common, "dataset"].nunique()),
    }
    return res


STEP = register(StepSpec(
    name="assemble_features",
    run=run,
    doc="Build X/y/groups for a target from chosen feature blocks (composition + modalities).",
    params_schema={
        "target": {"default": "subtype", "doc": "prediction target (see amlmm.targets.TARGETS)"},
        "blocks": {"default": None, "doc": "explicit feature blocks; None -> hooks.select_feature_blocks"},
        "min_class_n": {"default": 8, "doc": "min samples per class to keep it"},
    },
    consumes=["feasibility"],
    produces=["X", "y", "groups_donor", "groups_cohort", "target", "feature_blocks"],
))
