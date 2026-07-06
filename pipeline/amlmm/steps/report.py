"""Step: report — assemble a structured run report + markdown summary.

Collects coverage, every step's status/metrics, the feasibility table, the CV
result, and the gate decision into one json, then renders markdown via the
hooks.synthesize_report seam (deterministic template by default; an agent would
write a richer clinician-facing narrative here).
"""
from __future__ import annotations

from ..step import StepSpec, StepResult, register
from ..hooks import GateDecision


def run(ctx, params) -> StepResult:
    gate = ctx.getart("gate")
    gate_d = ({"accept": gate.accept, "reason": gate.reason, "action": gate.action}
              if isinstance(gate, GateDecision) else None)
    failing = [r.name for r in ctx.results if r.status == "error"]
    if failing:
        run_status = "errored"
    elif gate_d and gate_d["action"] == "abort":
        run_status = "aborted"
    else:
        run_status = "completed"
    run_report = {
        "run_id": ctx.config.run_id,
        "run_status": run_status,
        "failing_steps": failing,
        "provenance": ctx.getart("provenance"),
        "coverage": ctx.tables.get("coverage", {}),
        "target": ctx.getart("target"),
        "feature_blocks": ctx.getart("feature_blocks"),
        "steps": [r.to_dict() for r in ctx.results],
        "feasibility": ctx.getart("feasibility"),
        "cv": ctx.getart("cv_result"),
        "gate": gate_d,
    }
    md = ctx.hooks.synthesize_report(run_report)
    fpj = ctx.save_json(run_report, "run_report.json")
    fpm = ctx.path("REPORT.md")
    with open(fpm, "w", encoding="utf-8") as f:
        f.write(md)

    res = StepResult(name="report", artifacts={"run_report.json": fpj, "REPORT.md": fpm})
    res.metrics = {"report_json": fpj, "report_md": fpm}
    return res


STEP = register(StepSpec(
    name="report",
    run=run,
    doc="Assemble structured run_report.json + REPORT.md (markdown via synthesize_report seam).",
    consumes=["feasibility", "cv_result", "gate", "target"],
    produces=["run_report"],
))
