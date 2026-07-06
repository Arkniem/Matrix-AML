"""Agent roster — wraps the existing witness logic (panel.py) as registered AgentSpecs.

Phase A reuses three witnesses: genetic, cell-state/UDON, and predictive (composition or
any feature block). Domains are 'genetic' / 'cell_state' / 'predictive'; the grounding +
independence tags drive arbiter weighting. Each agent dispatches cohort vs patient on
`scope['mode']`, so one roster serves both. Imports panel lazily-safe: panel imports THIS
module only inside its run functions, so there is no import cycle.
"""
from __future__ import annotations
from ..agent import AgentSpec, register_agent
from .. import panel

MEASURED_BLOCKS = {"composition", "RNA"}


def _g_gather(ctx, scope):
    return (panel._patient_genetic(ctx, scope["sample_key"]) if scope["mode"] == "patient"
            else panel.evidence_genetic(ctx, scope["target"]))


def _g_assess(client, ctx, scope, ev):
    return (panel.assess_patient_genetic(client, ev) if scope["mode"] == "patient"
            else panel.assess_genetic(client, scope["target"], ev))


GENETIC = register_agent(AgentSpec("genetic", "genetic", "deterministic_fact", "independent",
                                   _g_gather, _g_assess, "mutations + targetable drivers"))


def _u_gather(ctx, scope):
    return (panel._patient_udon(ctx, scope["sample_key"]) if scope["mode"] == "patient"
            else panel.evidence_udon(ctx, scope["target"]))


def _u_assess(client, ctx, scope, ev):
    return (panel.assess_patient_udon(client, ev) if scope["mode"] == "patient"
            else panel.assess_udon(client, scope["target"], ev))


CELL_STATE = register_agent(AgentSpec("cell-state/UDON", "cell_state", "discovery", "rna_derived",
                                      _u_gather, _u_assess, "conserved UDON programs"))


def predictive_agent(block: str) -> AgentSpec:
    indep = "independent" if block in MEASURED_BLOCKS else "imputed_from_RNA"

    def gather(ctx, scope):
        st, perm = scope.get("strategy", "donor_kfold"), scope.get("permutations", 40)
        if scope["mode"] == "patient":
            return panel._patient_predictive(ctx, scope["sample_key"], block, st, perm)
        return panel.evidence_predictive(ctx, scope["target"], block, st, perm)

    def assess(client, ctx, scope, ev):
        if scope["mode"] == "patient":
            return panel.assess_patient_predictive(client, ev)
        return panel.assess_predictive(client, scope["target"], ev)

    return AgentSpec(block, "predictive", "honest_cv", indep, gather, assess,
                     f"predictive witness on {block}")


# ---- Phase B descriptive / corroborating witnesses ---------------------------------------
# Each uses a NEW domain string the arbiter's vote/anchor branches never read, so it can only
# contribute via the arbiter's additive harvest (therapy_biomarkers / validation_claims /
# descriptive_context) — it can never cast a subtype vote or move the genetic anchor. Patient
# mode only; cohort mode returns a skipped stub (the cohort path does not use this roster).
def _patient_only(gather_patient, assess_patient, witness, kind):
    def gather(ctx, scope):
        if scope.get("mode") == "patient":
            return gather_patient(ctx, scope["sample_key"])
        return {"witness": witness, "kind": kind, "status": "skipped",
                "reason": "descriptive witness runs in patient mode only"}

    def assess(client, ctx, scope, ev):
        if ev.get("status") != "ok":
            return {"confidence": 0.0, "reliability_weight": 0.0,
                    "summary": f"skipped ({ev.get('reason', '')})", "caveats": "skipped"}
        return assess_patient(client, ev)

    return gather, assess


def _reg(name, domain, grounding, independence, gatherer, assessor, doc):
    g, a = _patient_only(gatherer, assessor, name, domain)
    return register_agent(AgentSpec(name, domain, grounding, independence, g, a, doc))


LSC = _reg("LSC", "lsc", "classifier_call", "rna_derived",
           panel._patient_lsc, panel.assess_patient_lsc, "leukemic stem-cell architecture call")
SURFACEOME = _reg("surfaceome/ADT", "surfaceome", "descriptive_aggregate", "imputed_from_RNA",
                  panel._patient_surfaceome, panel.assess_patient_surfaceome,
                  "imputed surface markers -> flow/immunotherapy hypotheses")
METABOLIC = _reg("metabolic", "metabolic", "descriptive_aggregate", "imputed_from_RNA",
                 panel._patient_metabolic, panel.assess_patient_metabolic, "imputed metabolomics")
LIPID = _reg("lipid", "lipid", "descriptive_aggregate", "imputed_from_RNA",
             panel._patient_lipid, panel.assess_patient_lipid, "imputed lipidomics")
# NOTE: name is "GRN-regulon" (NOT "GRN") so it never collides with a predictive GRN block:
# two same-named specs would both append to the per-patient ledger and trip its immutability
# guard (different evidence) -> AssertionError aborts the run. default_roster also de-dups by name.
GRN = _reg("GRN-regulon", "grn", "descriptive_aggregate", "imputed_from_RNA",
           panel._patient_grn, panel.assess_patient_grn, "imputed gene-regulatory network (no fidelity)")
CELL_COMM = _reg("cell-communication", "cell_comm", "descriptive_aggregate", "independent",
                 panel._patient_signaling, panel.assess_patient_signaling,
                 "ligand-receptor signaling (independent axis)")

NEW_WITNESSES = {"lsc": LSC, "surfaceome": SURFACEOME, "metabolic": METABOLIC,
                 "lipid": LIPID, "grn": GRN, "cell_comm": CELL_COMM}


def default_roster(blocks=None, with_genetic=True, with_udon=True, *,
                   with_lsc=True, with_surfaceome=True,
                   descriptive=("metabolic", "lipid", "grn", "cell_comm")) -> list:
    # de-dup by witness name: the per-patient ledger keys on name, so two same-named specs
    # (e.g. a predictive 'GRN' block + a descriptive witness) would crash the run. First spec
    # for a given name wins (predictive blocks are added first).
    roster, used = [], set()

    def _add(spec):
        if spec.name not in used:
            roster.append(spec)
            used.add(spec.name)

    for b in (blocks or ["composition"]):
        _add(predictive_agent(b))
    if with_genetic:
        _add(GENETIC)
    if with_udon:
        _add(CELL_STATE)
    if with_lsc:
        _add(LSC)
    if with_surfaceome:
        _add(SURFACEOME)
    for d in descriptive:
        if d in NEW_WITNESSES:
            _add(NEW_WITNESSES[d])
    return roster
