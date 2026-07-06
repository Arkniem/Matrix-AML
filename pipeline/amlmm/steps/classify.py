"""Step: classify — honest grouped nested CV for the assembled target, then gate.

Pulls X/y/groups from the context, lets the hooks choose the model panel, runs
amlmm.cv.nested_cv_evaluate (donor-grouped or leave-one-cohort-out), gates the
result via hooks.gate_result (the skeptic seam), and fits a deployable final
model on all data. All reported numbers are out-of-fold.
"""
from __future__ import annotations
import numpy as np

from ..step import StepSpec, StepResult, register
from .. import models, cv


def run(ctx, params) -> StepResult:
    X = ctx.getart("X")
    y = ctx.getart("y")
    if X is None or y is None:
        return StepResult(name="classify", status="skipped",
                          metrics={"reason": "no features assembled (assemble_features skipped?)"})
    target = ctx.getart("target")
    feas = (ctx.getart("feasibility", {}) or {}).get(target, {})
    names = ctx.hooks.choose_models(target, feas)
    factories = models.build(names)

    result = cv.nested_cv_evaluate(
        X, y, ctx.getart("groups_donor"), ctx.getart("groups_cohort"),
        strategy=params["strategy"], model_factories=factories,
        outer_splits=int(params["outer_splits"]),
        inner_splits=int(params["inner_splits"]),
        n_permutations=int(params["n_permutations"]),
    )
    gate = ctx.hooks.gate_result(result)
    ctx.set("cv_result", result)
    ctx.set("gate", gate)
    fp = ctx.save_json(result, "cv_result.json")
    artifacts = {"cv_result.json": fp}

    res = StepResult(name="classify", artifacts=artifacts)
    final_path = None
    wins = result.get("model_wins")
    if result.get("error") or not wins or gate.action == "abort":
        res.add_log("skipped final-model fit: CV did not produce a trustworthy result "
                    f"(error={result.get('error')}, gate_action={gate.action})")
    else:
        try:
            import joblib
            # deterministic tie-break: most wins, then declaration order
            win = max(wins, key=lambda k: (wins[k], -names.index(k)))
            est, _ = models.build([win])[win]
            est.fit(np.asarray(X.values, dtype=float), np.asarray(y).astype(str))
            final_path = ctx.path("final_model.joblib")
            joblib.dump({"model": est, "features": list(X.columns),
                         "classes": sorted(set(np.asarray(y).astype(str))),
                         "winning_family": win, "target": target}, final_path)
            artifacts["final_model.joblib"] = final_path
        except Exception as e:
            res.add_log(f"final-model fit failed: {type(e).__name__}: {e}")

    res.metrics = {
        "target": target,
        "strategy": params["strategy"],
        "models": names,
        "balanced_accuracy": result.get("balanced_accuracy"),
        "macro_f1": result.get("macro_f1"),
        "n_samples": result.get("n_samples"),
        "n_classes": result.get("n_classes"),
        "permutation_p95": result.get("permutation_balanced_accuracy_p95"),
        "permutation_pvalue": result.get("permutation_pvalue"),
        "cv_error": result.get("error"),
        "model_wins": result.get("model_wins"),
        "gate_accept": gate.accept, "gate_reason": gate.reason, "gate_action": gate.action,
        "final_model": final_path,
    }
    return res


STEP = register(StepSpec(
    name="classify",
    run=run,
    doc="Honest group-aware nested CV for the assembled target + gate + deployable model.",
    params_schema={
        "strategy": {"default": "donor_kfold", "choices": ["donor_kfold", "leave_one_cohort_out"],
                     "doc": "outer CV grouping"},
        "outer_splits": {"default": 5, "doc": "outer folds (donor_kfold only)"},
        "inner_splits": {"default": 3, "doc": "inner folds for model selection"},
        "n_permutations": {"default": 100, "doc": "label permutations for the chance baseline"},
    },
    consumes=["X", "y", "groups_donor", "groups_cohort", "target", "feasibility"],
    produces=["cv_result", "gate"],
))
