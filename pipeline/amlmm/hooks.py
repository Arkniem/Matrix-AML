"""Decision hooks — the named seams where agents plug in.

The deterministic pipeline calls these at every judgment point. Today they use
simple, auditable defaults (`DecisionHooks`). To integrate agents, subclass and
override individual methods to consult an LLM (`AgentHooks`), returning the SAME
types. Nothing else in the pipeline changes -- the orchestrator just receives a
different hooks object. This is the "agent intervenes at the seam" design: the
agent owns read -> decide -> gate; the deterministic math is untouched.

Each method's contract (inputs, required return type) is documented so an agent
implementation has an exact target.
"""
from __future__ import annotations
from dataclasses import dataclass

# Canonical collapsing of the raw obs `Annotation` driver labels. Merging e.g.
# NPM1c into NPM1 is a genuine biological judgment call -> a natural agent seam
# (override `canonical_label_map`).
CANONICAL_DRIVER_MAP = {
    "NPM1c": "NPM1",
    "FLT3-ITD": "FLT3",
    "TP53 U2AF1": "TP53",
    "SF3B1 NPM1 FLT3": "multi",
    "CKIT": "KIT",          # align with the genetic anchor's "KIT" label
}

# Labels that are not specific genetic drivers (excluded from the driver task).
NON_DRIVER_LABELS = {"AML", "Pediatric-AML", "Control", "nan", "0", ""}


@dataclass
class GateDecision:
    accept: bool
    reason: str
    action: str = "pass"          # pass | rerun | abort


class DecisionHooks:
    """Auditable deterministic defaults. Safe to use as-is for a baseline run."""
    name = "deterministic"

    # --- seam 1: label canonicalization -------------------------------------
    def canonical_label_map(self, annotations) -> dict:
        """annotations: iterable[str] of raw obs Annotation values.
        return: dict raw -> canonical label. AgentHooks could revise merges
        using genetics knowledge (e.g. whether to split FLT3-ITD vs FLT3-TKD)."""
        return {a: CANONICAL_DRIVER_MAP.get(a, a) for a in set(annotations)}

    # --- seam 2: feature-block selection ------------------------------------
    def select_feature_blocks(self, target, available_blocks, feasibility) -> list:
        """target: str. available_blocks: list[str] (e.g. composition, RNA,
        GRN, Metabolite, Lipid, ADT). feasibility: dict for this target.
        return: ordered list[str] of blocks to assemble. Default = composition
        only (fully measured, proven strongest). AgentHooks could add modality
        blocks or run an ablation."""
        return ["composition"] if "composition" in available_blocks else available_blocks[:1]

    # --- seam 3: model choice -----------------------------------------------
    def choose_models(self, target, feasibility) -> list:
        """return: list[str] of model keys from amlmm.models.MODEL_ZOO.
        Default = a robust RF + regularized logistic panel."""
        return ["rf", "logreg"]

    # --- seam 4: result gating (the skeptic) --------------------------------
    def gate_result(self, cv_result) -> GateDecision:
        """cv_result: dict from cv.nested_cv_evaluate. Decide whether the result is
        trustworthy enough to pass downstream. Primary criterion is the SELF-CONSISTENT
        group-level permutation p-value (reference estimator on fixed folds); falls back
        to the p95 comparison only if no p-value. Low-n flagged provisional.
        AgentHooks runs an adversarial LLM skeptic here."""
        if cv_result.get("error"):
            return GateDecision(False, str(cv_result["error"]), "abort")
        ba = cv_result.get("balanced_accuracy")
        pval = cv_result.get("permutation_pvalue")
        chance = cv_result.get("permutation_balanced_accuracy_p95")
        n = int(cv_result.get("n_samples", 0))
        if ba is None:
            return GateDecision(False, "no metric produced", "abort")
        if pval is not None:
            if pval >= 0.05:
                return GateDecision(False, f"permutation p={pval:.3f}: signal not above chance under grouped CV", "rerun")
            note = f"balanced_acc {ba:.3f}, permutation p={pval:.3f}"
        elif chance is not None and ba <= chance:
            return GateDecision(False, f"balanced_acc {ba:.3f} <= permutation p95 {chance:.3f}", "rerun")
        else:
            note = f"balanced_acc {ba:.3f}"
        if n < 40:
            return GateDecision(True, f"{note}; low n={n}, provisional", "pass")
        return GateDecision(True, f"{note} (n={n})", "pass")

    # --- seam 6: discovery per-combo feature selection ----------------------
    def discovery_feature_select(self, field, modality, cell_state, n_features) -> dict:
        """Per-(field x modality x cell-state) SUPERVISED feature-selection choice for the Discovery
        agent. `n_features` = width AFTER the unsupervised variance prefilter. Return
        {"selector": "f_classif"|"mutual_info"|"none", "k": int} to override, or {} to use the
        DiscoveryConfig defaults. Default = {} (use config). An AgentHooks could pick mutual_info for a
        modality where ANOVA misses interactions, widen k for a dense modality, or disable selection for
        an already-narrow one. The selector still runs IN-FOLD (leakage-safe) regardless of choice."""
        return {}

    # --- seam 5: report synthesis (the synthesizer) -------------------------
    def synthesize_report(self, run_report) -> str:
        """run_report: dict (assembled by the report step). return: markdown str.
        Default = a deterministic template. AgentHooks would write a richer,
        clinician-facing narrative with caveats."""
        lines = [f"# AML multimodal pipeline run: {run_report.get('run_id', '?')}", ""]
        cov = run_report.get("coverage", {})
        lines.append(f"- samples: {cov.get('n_samples')} | pseudobulks: {cov.get('n_pseudobulks')} "
                     f"| modalities: {', '.join(cov.get('modalities', []))}")
        for step in run_report.get("steps", []):
            lines.append(f"- step `{step['name']}` [{step['status']}, {step['seconds']}s]: "
                         f"{', '.join(f'{k}={v}' for k, v in list(step.get('metrics', {}).items())[:6])}")
        gate = run_report.get("gate")
        if gate:
            lines += ["", f"**Gate:** {'ACCEPT' if gate['accept'] else 'REJECT'} — {gate['reason']} "
                          f"(action={gate['action']})"]
        return "\n".join(lines)


class AgentHooks(DecisionHooks):
    """LLM-backed hooks: each decision is made by the configured model (default
    nemotron-3-super via the CCHMC gateway, see amlmm/llm.py). Every method
    validates the model's reply and, on any error/invalid output, falls back to the
    deterministic default from the superclass -- so a gateway outage degrades to the
    auditable baseline rather than crashing the deterministic core. `canonical_label_map`
    is intentionally NOT delegated (label merging stays deterministic/auditable)."""
    name = "agent"

    def __init__(self, client=None):
        from .llm import LLMClient
        self.client = client or LLMClient()

    def gate_result(self, cv_result) -> GateDecision:
        import json
        from .llm import LLMError
        if cv_result.get("error"):
            return GateDecision(False, str(cv_result["error"]), "abort")
        keys = ("balanced_accuracy", "macro_f1", "per_class_f1", "class_counts",
                "n_samples", "n_classes", "permutation_balanced_accuracy_p95",
                "permutation_pvalue", "balanced_accuracy_fold_std", "strategy", "model_wins")
        summary = {k: cv_result.get(k) for k in keys}
        prompt = ("You are a skeptical biostatistician reviewing a DONOR-GROUPED cross-validation "
                  "result from an AML genetic-subtype classifier. Decide if it is trustworthy enough "
                  "to pass downstream. Be adversarial: reject (action 'rerun') if the permutation "
                  "p-value is >= 0.05, if n is tiny, if one class dominates, or if fold variance is high; "
                  "abort if the result is degenerate. "
                  f"Result: {json.dumps(summary, default=str)}. "
                  'Return ONLY JSON {"accept": bool, "reason": str, "action": "pass"|"rerun"|"abort"}.')
        try:
            o = self.client.chat_json(prompt, required=("accept", "reason", "action"))
            act = o["action"] if o.get("action") in ("pass", "rerun", "abort") else "pass"
            return GateDecision(bool(o["accept"]), str(o["reason"])[:500], act)
        except (LLMError, Exception):
            return DecisionHooks.gate_result(self, cv_result)

    def select_feature_blocks(self, target, available_blocks, feasibility) -> list:
        import json
        from .llm import LLMError
        prompt = ("Choose which feature blocks to use for a classifier. "
                  f"target={target}. available={available_blocks} "
                  "(composition = MEASURED cell-type frequencies; the others are imputed-FROM-RNA "
                  "modalities -- lower fidelity, NOT independent evidence). Prefer measured composition; "
                  "add a modality only to run a deliberate ablation. "
                  f"feasibility={json.dumps(feasibility, default=str)[:600]}. "
                  'Return ONLY JSON {"blocks": [..]} as a non-empty subset of available.')
        try:
            o = self.client.chat_json(prompt, required=("blocks",))
            blocks = [b for b in o["blocks"] if b in available_blocks]
            return blocks or DecisionHooks.select_feature_blocks(self, target, available_blocks, feasibility)
        except (LLMError, Exception):
            return DecisionHooks.select_feature_blocks(self, target, available_blocks, feasibility)

    def choose_models(self, target, feasibility) -> list:
        from .llm import LLMError
        from .models import MODEL_ZOO
        avail = list(MODEL_ZOO)
        prompt = (f"Choose ML models for target={target} from {avail} (rf=random forest, "
                  "logreg=regularized logistic, gboost=hist gradient boosting). "
                  'Return ONLY JSON {"models": [..]} as a non-empty subset of the listed keys.')
        try:
            o = self.client.chat_json(prompt, required=("models",))
            ms = [m for m in o["models"] if m in MODEL_ZOO]
            return ms or DecisionHooks.choose_models(self, target, feasibility)
        except (LLMError, Exception):
            return DecisionHooks.choose_models(self, target, feasibility)

    def synthesize_report(self, run_report) -> str:
        import json
        from .llm import LLMError
        prompt = ("Write a concise, candid, clinician-facing markdown summary of this AML multimodal "
                  "pipeline run. State the target, the honest out-of-fold balanced accuracy vs the "
                  "group-level permutation p-value, and the gate decision. Include explicit caveats: "
                  "imputed-from-RNA modalities are not independent evidence; small n; donor/cohort-grouped CV. "
                  f"Run report JSON: {json.dumps(run_report, default=str)[:4000]}")
        try:
            return self.client.chat(prompt, json_mode=False)
        except (LLMError, Exception):
            return DecisionHooks.synthesize_report(self, run_report)
