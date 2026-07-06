"""Phase C — guarded continuous feedback / blackboard deliberation.

Round 0 is the INDEPENDENT baseline (each witness's opinion formed alone), recorded
immutably in the ledger. In rounds 1..max_rounds each VOTING witness reads the arbiter's
previous-round consensus and DETERMINISTICALLY revises its opinion: a predictive or
cell-state witness whose round-0 vote CONFLICTS with a *genetically-confirmed* leading
hypothesis DEFERS — it down-weights itself toward the observed driver (an imputed / RNA-
derived prediction should not argue with a directly observed mutation). The revision is
deterministic and applied once per witness, so the loop converges and is reproducible.

Guardrails that make deliberation safe (cannot override ground truth):
  * EVIDENCE is immutable — only the `opinion` (a witness's confidence/weight) is revised;
    the ledger's append guard enforces this, and the deterministic_evidence_hash is unchanged.
  * The genetic ANCHOR is computed from evidence (not opinions) every round, so a present
    driver ALWAYS leads — even if every witness converged on a different subtype.
  * The round-0 baseline consensus is preserved; `drift` diffs the final vs the baseline and
    raises a groupthink_warning ONLY if the leading hypothesis changed AND the final is not
    genetically anchored (the one case worth a human's eye).
  * `max_rounds` caps the loop; convergence (stable leading + concordance + consistency) ends
    it early. `mode='conflict_triggered'` skips deliberation entirely on a concordant round 0
    (kept as a comparison against the figure-literal 'continuous' mode).

What deliberation can and cannot change: it can refine CONCORDANCE / confidence and the
per-witness narrative (a conflicting imputed witness defers, so concordance rises); it can
NEVER change the leading hypothesis, the driver therapies, or the anchor.
"""
from __future__ import annotations

from .agent import AgentResult
from . import arbiter

VOTING_DOMAINS = ("predictive", "cell_state")   # genetic anchors (never defers); descriptive don't vote


def _revise(entry: dict, prev: dict, deference: float, rnd: int) -> dict:
    """Deterministic opinion revision for one witness given the previous consensus."""
    op = dict(entry.get("opinion") or {})
    if entry.get("domain") not in VOTING_DOMAINS:
        return op                                  # genetic = ground truth; descriptive = non-voting
    if not (prev or {}).get("leading_confirmed_by_genetics"):
        return op                                  # no confirmed anchor -> the votes ARE the signal
    consistency = (prev or {}).get("per_witness_consistency", {}) or {}
    if consistency.get(entry["witness"]) == "conflict" and not op.get("revised"):
        w = float(op.get("reliability_weight") or 0.0)
        op["reliability_weight"] = round(w * deference, 3)
        op["revised"] = True
        op["caveats"] = (str(op.get("caveats", "")) +
                         f" [r{rnd}: deferred to observed driver "
                         f"{prev.get('leading_hypothesis')}]").strip()
    return op


def _converged(cur: dict, prev: dict) -> bool:
    return (cur.get("leading_hypothesis") == prev.get("leading_hypothesis")
            and round(float(cur.get("concordance") or 0.0), 6)
                == round(float(prev.get("concordance") or 0.0), 6)
            and cur.get("per_witness_consistency") == prev.get("per_witness_consistency"))


def _drift(baseline: dict, final: dict) -> dict:
    b, f = baseline or {}, final or {}
    bl, fl = b.get("leading_hypothesis"), f.get("leading_hypothesis")
    changed = (bl != fl)
    confirmed = bool(f.get("leading_confirmed_by_genetics"))
    return {
        "baseline_leading": bl, "final_leading": fl, "leading_changed": changed,
        "final_genetically_confirmed": confirmed,
        "groupthink_warning": bool(changed and not confirmed),
        "baseline_concordance": b.get("concordance"), "final_concordance": f.get("concordance"),
    }


def deliberate(ctx, ledger, client, max_rounds: int = 2, mode: str = "continuous",
               deference: float = 0.5) -> dict:
    """Run the guarded feedback loop. Assumes round 0 (witness entries + arbiter consensus)
    is already in the ledger. Finalizes the ledger and returns the converged consensus, the
    round-0 baseline, the stop reason, and the groupthink-drift report."""
    baseline = ledger.arbiter_consensus(0)

    def _result(consensus, stop_reason):
        # one canonical stop_reason for BOTH the persisted ledger and the returned report,
        # and a deliberation-round count that excludes the round-0 baseline.
        ledger.finalize(stop_reason)
        return {"consensus": consensus, "baseline": baseline, "mode": mode,
                "stop_reason": stop_reason,
                "rounds_run": ledger.data["rounds_run"],                    # ledger rounds incl. baseline
                "deliberation_rounds": max(0, ledger.data["rounds_run"] - 1),  # rounds BEYOND baseline
                "drift": _drift(baseline, consensus)}

    if int(max_rounds) < 1:                                  # direct-call sanity (panel also guards)
        return _result(baseline, "single_pass")
    if mode == "conflict_triggered" and not (baseline or {}).get("conflicts"):
        return _result(baseline, "no_conflict")              # concordant round 0 -> no deliberation

    prev = baseline
    stop_reason = "max_rounds"
    for r in range(1, int(max_rounds) + 1):
        revisions = []
        for e in ledger.current_entries():
            new_op = _revise(e, prev, deference, r)
            if new_op != (e.get("opinion") or {}):
                revisions.append((e, new_op))
        if not revisions:                                   # nobody changed -> stable
            stop_reason = "converged"
            break
        for e, new_op in revisions:
            ledger.append(AgentResult(name=e["witness"], domain=e["domain"],
                                      grounding=e["grounding"], independence=e["independence"],
                                      status=e.get("status", "ok"), error=e.get("error"),
                                      evidence=e["evidence"], opinion=new_op), round=r)
        cur = arbiter.reconcile_patient(client, ctx, ledger)
        ledger.set_arbiter(cur, round=r)
        if _converged(cur, prev):
            stop_reason = "converged"
            prev = cur
            break
        prev = cur

    return _result(prev, stop_reason)
