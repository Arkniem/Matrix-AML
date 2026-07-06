"""Arbiter v2 — therapy-board reconciliation with a DETERMINISTIC pre-pass.

The pre-pass (not the LLM) makes the decision, so it is reproducible and honest:
  * each witness's subtype signal is weighted by reliability × grounding × independence
    × CV-gate (an imputed, below-chance predictor counts for little);
  * the GENETIC ANCHOR: if the genetic witness reports a *present* driver mutation, that
    driver's subtype is the leading hypothesis — a single imputed prediction that disagrees
    is recorded as a conflict but CANNOT outrank it (this is the TP53 fix);
  * therapies are drawn ONLY from the curated knowledge base, keyed on OBSERVED genetic
    drivers (not on an imputed subtype guess); a predicted-but-unconfirmed subtype drives a
    *validation* recommendation, not a direct therapy;
  * validations come from the KB validation rules for the present claim types.
The LLM only narrates a rationale + sets confidence within these fixed conclusions; on any
LLM error the deterministic pre-pass IS the returned consensus.
"""
from __future__ import annotations
import json

from .agent import GROUNDING_FACTOR, INDEPENDENCE_FACTOR
from .llm import LLMError

# de-prefixed genetic flag -> subtype/lesion label. Every cytogenetic flag that
# genetics.build_mutation_matrix can emit is mapped, so an OBSERVED lesion always
# anchors and can never be outranked by an imputed prediction (the design intent).
# KIT -> "KIT" (not "CKIT") so the anchor label matches the classifier label space
# (hooks.CANONICAL_DRIVER_MAP collapses the raw "CKIT" annotation to "KIT").
ANCHOR_MAP = {"NPM1": "NPM1", "FLT3": "FLT3", "TP53": "TP53", "IDH1": "IDH1", "IDH2": "IDH2",
              "inv16": "Inv16", "del7": "DEL(7)", "TET2": "TET2", "SF3B1": "SF3B1",
              "SRSF2": "SRSF2", "KIT": "KIT", "NRAS": "NRAS", "KRAS": "KRAS", "WT1": "WT1",
              "CEBPA": "CEBPA", "DNMT3A": "DNMT3A", "ASXL1": "ASXL1", "RUNX1": "RUNX1",
              "CBL": "CBL", "CSF3R": "CSF3R", "U2AF1": "U2AF1",
              "t15_17": "APL", "t8_21": "t(8;21)", "kmt2a": "KMT2Ar", "del5": "DEL(5)",
              "trisomy8": "Trisomy8", "complex": "Complex"}
# APL/t(15;17) first: a therapy-defining emergency (ATRA/ATO) that must never be
# overridden. Cytogenetic + molecular drivers interleaved by clinical weight.
ANCHOR_PRIORITY = ["t15_17", "TP53", "FLT3", "NPM1", "kmt2a", "inv16", "t8_21", "IDH1", "IDH2",
                   "KIT", "RUNX1", "TET2", "SF3B1", "SRSF2", "ASXL1", "DNMT3A", "NRAS", "KRAS",
                   "WT1", "CEBPA", "complex", "del7", "del5", "trisomy8", "CBL", "CSF3R", "U2AF1"]
# de-prefixed cytogenetic flags whose presence is itself a claim worth a tailored
# validation (mirrors the claim_types built in reconcile_patient).
CYTO_FLAGS = {"inv16", "del7", "del5", "trisomy8", "complex", "t8_21", "t15_17", "kmt2a"}
# non-actionable / generic subtype tokens that must never be the leading hypothesis
# when no driver anchors (e.g. the merged "multi" class from CANONICAL_DRIVER_MAP).
GENERIC_SUBTYPES = {"multi", "AML", "Control", "Pediatric-AML", "nan", "", None}
# validation claim_types a descriptive witness may inject via the additive harvest. Genetic
# claim_types (subtype/mutation/cytogenetics) come ONLY from the anchored genetic path, so a
# witness cannot conjure a FISH/sequencing recommendation for a lesion the patient lacks.
HARVESTABLE_CLAIMS = {"lsc", "surface_marker", "metabolite", "lipid_profile"}


def _eff_weight(entry):
    op = entry.get("opinion", {}) or {}
    w = float(op.get("reliability_weight") or 0.0)
    w *= GROUNDING_FACTOR.get(entry.get("grounding"), 0.5)
    w *= INDEPENDENCE_FACTOR.get(entry.get("independence"), 0.5)
    return w


def _above_chance(ev):
    p = ev.get("cohort_permutation_p")
    if p is None:
        p = ev.get("permutation_pvalue")
    return (p is not None and p < 0.05)


def reconcile_patient(client, ctx, ledger) -> dict:
    kb = ctx.knowledge
    # latest opinion per witness (= round 0 in single-pass; the revised round in Phase C
    # deliberation) — never double-counts a witness across feedback rounds.
    entries = ledger.current_entries() if hasattr(ledger, "current_entries") else ledger.entries()
    signals = []                 # (witness, subtype, weight) -- ONLY genetic/predictive/cell_state vote
    present, targetable, anchor = [], {}, None
    extra_biomarkers, extra_claims, descriptive = [], [], []   # additive, non-voting harvest

    for e in entries:
        dom, ev = e.get("domain"), e.get("evidence", {}) or {}
        w = _eff_weight(e)
        if dom == "genetic":
            present = ev.get("present", []) or []
            targetable = ev.get("targetable", {}) or {}
            for drv in sorted(present, key=lambda d: ANCHOR_PRIORITY.index(d)
                              if d in ANCHOR_PRIORITY else 99):
                if drv in ANCHOR_MAP:
                    anchor = ANCHOR_MAP[drv]
                    break
            if anchor:
                signals.append((e["witness"], anchor, w))
        elif dom == "predictive":
            pred = ev.get("patient_prediction")
            w *= (1.0 if _above_chance(ev) else 0.3)
            if pred:
                signals.append((e["witness"], pred, w))
        elif dom == "cell_state":
            marks = [a.get("marks") for a in (ev.get("active_programs") or []) if a.get("marks")]
            if marks:
                signals.append((e["witness"], marks[0], w))

        # generic additive harvest from EVERY entry (incl. the descriptive Phase B witnesses).
        # These feed therapies/validations/context but NEVER the subtype vote or the anchor.
        extra_biomarkers += [str(b) for b in (ev.get("therapy_biomarkers") or [])]
        extra_claims += [str(c) for c in (ev.get("validation_claims") or [])]
        for s in (ev.get("descriptive_context") or []):
            descriptive.append({"witness": e.get("witness"), "domain": dom, "note": str(s)[:240]})

    support = {}
    for _, st, w in signals:
        support[st] = support.get(st, 0.0) + w
    if anchor:
        leading = anchor
    else:
        cand = {k: v for k, v in support.items() if k not in GENERIC_SUBTYPES}
        leading = (max(cand, key=cand.get) if cand
                   else (max(support, key=support.get) if support else None))
    total = sum(support.values()) or 1.0
    concordance = round((support.get(leading, 0.0) / total) if leading else 0.0, 3)

    consistency, conflicts = {}, []
    for wn, st, _w in signals:
        agree = (st == leading)
        consistency[wn] = "agree" if agree else "conflict"
        if not agree:
            conflicts.append(f"{wn}→{st}")

    # therapies: keyed on OBSERVED drivers + cytogenetics (never on an imputed guess)
    biomarkers = list(present)
    therapies = kb.therapies_for(biomarkers) if kb else []
    leading_confirmed = bool(anchor) and ANCHOR_MAP.get(
        next((d for d in present if ANCHOR_MAP.get(d) == leading), ""), None) == leading

    # surface-marker therapy HYPOTHESES from observed-elevated (imputed) surface markers: kept
    # SEPARATE from driver-anchored therapies and flagged flow-pending; they never anchor/vote.
    surface_biomarkers = sorted(set(extra_biomarkers))
    surface_therapies = [{**t, "requires_flow_confirmation": True}
                         for t in (kb.therapies_for(surface_biomarkers)
                                   if (kb and surface_biomarkers) else [])]

    claim_types = ["subtype"]
    if present:
        claim_types.append("mutation")
    for cf in sorted(CYTO_FLAGS):
        if cf in present:
            claim_types.append(cf)
    if leading and not leading_confirmed:
        claim_types.append("mutation")     # confirm the predicted subtype by sequencing
    claim_types += sorted(set(extra_claims) & HARVESTABLE_CLAIMS)   # descriptive claims only
    validations = kb.validations_for(sorted(set(claim_types))) if kb else []
    descriptive.sort(key=lambda d: (str(d.get("witness")), str(d.get("note"))))   # stable

    if anchor and concordance >= 0.6:
        confidence = "high"
    elif anchor or concordance >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    anchor_note = (" (genetic-anchored: direct mutation observation)" if anchor
                   else " (witness consensus)")
    rationale = (f"Leading hypothesis: {leading}{anchor_note}. Concordance {concordance:.2f}. "
                 + (f"Conflicts: {', '.join(conflicts)}. " if conflicts else "")
                 + ("Therapies follow observed drivers; a predicted subtype must be confirmed by "
                    "sequencing before subtype-directed therapy." if not leading_confirmed and leading else ""))

    consensus = {
        "subtype_call": (f"{leading}{anchor_note}" if leading else "indeterminate"),
        "leading_hypothesis": leading,
        "leading_confirmed_by_genetics": bool(leading_confirmed),
        "concordance": concordance,
        "per_witness_consistency": consistency,
        "conflicts": "; ".join(conflicts),
        "ranked_therapy_hypotheses": therapies,
        "surface_therapy_hypotheses": surface_therapies,
        "recommended_validations": validations,
        "descriptive_findings": descriptive,
        "overall_confidence": confidence,
        "rationale": rationale,
        "knowledge_version": (kb.version if kb else None),
    }

    # LLM narration, constrained: it may rewrite rationale + set confidence, never the decision
    if client is not None:
        try:
            brief = {"leading_hypothesis": leading, "anchored": bool(anchor),
                     "signals": [{"witness": w, "subtype": s, "weight": round(wt, 3)} for w, s, wt in signals],
                     "therapies": therapies, "validations": validations, "conflicts": conflicts}
            out = client.chat_json(
                "You are the AML tumor-board chair writing the rationale for a per-patient decision. "
                f"The deterministic pre-pass already DECIDED: {json.dumps(brief, default=str)[:2500]}. "
                "Write a concise clinician rationale and set overall_confidence. You MUST NOT change the "
                "leading hypothesis, the therapies, or add therapies/validations — only explain and "
                "justify, honoring the genetic anchor. "
                'Return ONLY JSON {"rationale": str, "overall_confidence": "low"|"medium"|"high"}.',
                required=("rationale", "overall_confidence"))
            consensus["rationale"] = str(out.get("rationale", rationale))[:1200]
            lvl = out.get("overall_confidence")
            rank = {"low": 0, "medium": 1, "high": 2}
            if lvl in rank:
                # Honesty guard: the LLM may REFINE confidence (incl. lower it), but it may not
                # RAISE it above the deterministic ceiling for a call that is NOT genetically
                # confirmed (no observed-driver anchor). Prevents the narrator inflating an
                # unanchored, imputed/predicted hypothesis (e.g. a prob-0.4 subtype guess) to "high".
                if not leading_confirmed and rank[lvl] > rank.get(consensus["overall_confidence"], 1):
                    lvl = consensus["overall_confidence"]
                consensus["overall_confidence"] = lvl
        except (LLMError, Exception):
            pass
    return consensus


def reconcile_cohort(client, ctx, ledger) -> dict:
    """Cohort-level summary: which witnesses carry signal + the targetable landscape +
    KB therapies for the cohort's present drivers. Deterministic pre-pass + optional narration."""
    kb = ctx.knowledge
    entries = ledger.entries()
    above, consistency, targetable = [], {}, {}
    for e in entries:
        dom, ev = e.get("domain"), e.get("evidence", {}) or {}
        if dom == "predictive":
            consistency[e["witness"]] = "above_chance" if _above_chance(ev) else "at_chance"
            if _above_chance(ev):
                above.append(e["witness"])
        elif dom == "genetic":
            # cohort genetic evidence comes from summarize_genetics, whose key is
            # `targetable_present` (the per-patient path uses `targetable`).
            targetable = ev.get("targetable_present", ev.get("targetable", {})) or {}
            consistency[e["witness"]] = "independent"
        elif dom == "cell_state":
            consistency[e["witness"]] = "discovery"
    therapies = kb.therapies_for(list(targetable.keys())) if kb else []
    validations = kb.validations_for(["subtype", "mutation"]) if kb else []
    consensus = {
        "above_chance_witnesses": above,
        "targetable_landscape": targetable,
        "ranked_therapy_hypotheses": therapies,
        "recommended_validations": validations,
        "per_witness_consistency": consistency,
        "overall_confidence": "medium" if above else "low",
        "knowledge_version": (kb.version if kb else None),
        "rationale": (f"Witnesses above chance: {above or 'none'}. "
                      f"Targetable drivers present: {list(targetable)}."),
    }
    return consensus
