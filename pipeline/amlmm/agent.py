"""Agent registry — the witness analogue of step.py.

Each expert agent is an `AgentSpec` with a `gather` (deterministic evidence read) and
an `assess` (LLM interpretation with deterministic fallback), plus two fields the
arbiter weights on: `grounding` (how trustworthy the evidence is) and `independence`
(is it genuinely independent of RNA, or RNA-derived/imputed). `run_agent` mirrors
`run_step`: it times the call, captures errors, and returns an `AgentResult`. The
deterministic core (cv/genetics/udon) is what `gather` reads — agents never recompute it.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable

# how strongly the arbiter trusts each grounding / independence tier
GROUNDING_FACTOR = {"deterministic_fact": 1.0, "honest_cv": 0.9,
                    "classifier_call": 0.7, "descriptive_aggregate": 0.5}
INDEPENDENCE_FACTOR = {"independent": 1.0, "rna_derived": 0.6,
                       "imputed_from_RNA": 0.5, "discovery": 0.7}


@dataclass
class AgentResult:
    name: str
    domain: str
    grounding: str
    independence: str
    status: str = "ok"                       # ok | error | skipped
    evidence: dict = field(default_factory=dict)
    opinion: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "domain": self.domain, "grounding": self.grounding,
                "independence": self.independence, "status": self.status,
                "evidence": self.evidence, "opinion": self.opinion,
                "seconds": self.seconds, "error": self.error}


@dataclass
class AgentSpec:
    name: str
    domain: str
    grounding: str            # deterministic_fact | honest_cv | classifier_call | descriptive_aggregate
    independence: str         # independent | rna_derived | imputed_from_RNA | discovery
    gather: Callable          # (ctx, scope) -> evidence dict
    assess: Callable          # (client, ctx, scope, evidence) -> opinion dict
    doc: str = ""


REGISTRY: dict[str, AgentSpec] = {}


def register_agent(spec: AgentSpec) -> AgentSpec:
    REGISTRY[spec.name] = spec
    return spec


def get_agent(name: str) -> AgentSpec:
    return REGISTRY[name]


def list_agents() -> list[AgentSpec]:
    return list(REGISTRY.values())


def run_agent(spec: AgentSpec, ctx, scope: dict, client) -> AgentResult:
    """Gather grounded evidence then assess it. scope = {'mode':'patient','sample_key':..}
    or {'mode':'cohort','target':..} (+ optional strategy/permutations)."""
    t0 = time.perf_counter()
    res = AgentResult(name=spec.name, domain=spec.domain,
                      grounding=spec.grounding, independence=spec.independence)
    try:
        res.evidence = spec.gather(ctx, scope) or {}
        res.status = res.evidence.get("status", "ok")
        res.opinion = spec.assess(client, ctx, scope, res.evidence) or {}
    except Exception as e:
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
    res.seconds = round(time.perf_counter() - t0, 2)
    return res
