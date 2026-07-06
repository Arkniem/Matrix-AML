"""amlmm — AML multimodal base pipeline.

A deterministic, reproducible pipeline over the local AML multimodal atlas
(RNA + imputed GRN/metabolite/lipid/ADT + cell-communication + UDON clusters +
clinical labels), structured so that LLM agents can later be slotted in at named
decision points without touching the deterministic core.

Two layers, deliberately separated (see docs/AGENT_INTEGRATION.md):
  * Deterministic core  — data IO, the leakage-proof CV harness, the model zoo,
    and the step mechanics. Reproducible, auditable, identical every run.
  * Decision hooks       — the "seams". The orchestrator calls these at every
    judgment point (feature/model choice, result gating, report synthesis).
    Today they use auditable defaults (DecisionHooks); swap in AgentHooks to let
    agents drive, with the core untouched.
"""
from .context import Config, Context, build_context
from .step import StepSpec, StepResult, register, run_step, list_steps, REGISTRY
from .hooks import DecisionHooks, AgentHooks, GateDecision

__all__ = [
    "Config", "Context", "build_context",
    "StepSpec", "StepResult", "register", "run_step", "list_steps", "REGISTRY",
    "DecisionHooks", "AgentHooks", "GateDecision",
]
__version__ = "0.1.0"
