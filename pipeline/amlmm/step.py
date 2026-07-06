"""Step abstraction + registry — the unit an agent will later drive.

A step is a function `run(ctx, params) -> StepResult`. A StepSpec wraps it with a
name, an agent/human-readable doc, a params schema (the "knobs"), and declared
consumes/produces artifact keys. The registry lets an orchestrator -- deterministic
now, agent-driven later -- enumerate steps, read their schemas, invoke them with
chosen params, and branch on the returned StepResult.metrics (the "readout").
That read -> decide -> act -> gate loop is the seam in docs/AGENT_INTEGRATION.md.
"""
from __future__ import annotations
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StepResult:
    name: str
    status: str = "ok"                              # ok | error | skipped
    metrics: dict = field(default_factory=dict)     # json-serializable readout
    artifacts: dict = field(default_factory=dict)   # name -> path written on disk
    log: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None

    def add_log(self, msg: str) -> None:
        self.log.append(str(msg))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "metrics": self.metrics,
            "artifacts": self.artifacts, "params": self.params,
            "seconds": self.seconds, "error": self.error, "log": self.log,
        }


@dataclass
class StepSpec:
    name: str
    run: Callable[..., "StepResult | dict | None"]
    doc: str = ""
    params_schema: dict = field(default_factory=dict)   # param -> {default, doc, choices?}
    consumes: list = field(default_factory=list)         # artifact keys read from ctx
    produces: list = field(default_factory=list)         # artifact keys written to ctx

    def default_params(self) -> dict:
        return {k: v.get("default") for k, v in self.params_schema.items()}


REGISTRY: dict[str, StepSpec] = {}


def register(spec: StepSpec) -> StepSpec:
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> StepSpec:
    return REGISTRY[name]


def list_steps() -> list[StepSpec]:
    return list(REGISTRY.values())


def run_step(spec: StepSpec, ctx, params: dict | None = None) -> StepResult:
    """Invoke a step with merged params, capturing timing/errors into StepResult."""
    p = spec.default_params()
    p.update(params or {})
    res = StepResult(name=spec.name, params=p)
    t0 = time.perf_counter()
    try:
        out = spec.run(ctx, p)
        if isinstance(out, StepResult):
            res = out
            res.params = p
        elif isinstance(out, dict):
            res.metrics = out
    except Exception as e:
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
        res.add_log(traceback.format_exc())
    res.seconds = round(time.perf_counter() - t0, 2)
    ctx.record(res)
    return res
