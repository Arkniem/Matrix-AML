"""Shared evidence ledger (blackboard) — per-patient (or per-cohort-run).

Append-only record of every agent's contribution: the grounded `evidence` (IMMUTABLE)
and the agent's `opinion` (revisable across rounds). Persisted via ctx.save_json after
every change. A `deterministic_evidence_hash` lets a re-run confirm the grounded inputs
were identical even when LLM prose differs (separates "did the science change" from
"did the wording change"). In Phase A this is single-round; the read API + immutability
are what later feedback rounds (Phase C) build on.
"""
from __future__ import annotations
import hashlib
import json


class Ledger:
    def __init__(self, scope_id: str, target: str, provenance: dict | None = None):
        self.data = {
            "schema_version": 1,
            "scope_id": scope_id,            # sample_key (patient) or target (cohort)
            "target": target,
            "provenance": provenance,
            "entries": [],                   # one per agent per round
            "arbiter_rounds": [],
            "rounds_run": 0,
            "stop_reason": "single_pass",
        }

    def append(self, result, round: int = 0, read_ids=None) -> None:
        """Append an agent's entry. Evidence is recorded immutably; if the same witness
        already has an earlier-round entry, the evidence block must match (guards against
        an agent silently rewriting the grounded numbers)."""
        prior = self.latest(result.name)
        if prior is not None and prior["evidence"] != result.evidence:
            raise AssertionError(f"ledger: {result.name} tried to mutate its evidence block")
        self.data["entries"].append({
            "entry_id": f"{result.name}#r{round}",
            "witness": result.name, "domain": result.domain,
            "grounding": result.grounding, "independence": result.independence,
            "round": round, "status": result.status, "error": result.error,
            "evidence": result.evidence, "opinion": result.opinion,
            "read_entry_ids": read_ids or [],
        })
        self.data["rounds_run"] = max(self.data["rounds_run"], round + 1)

    def latest(self, witness: str):
        hits = [e for e in self.data["entries"] if e["witness"] == witness]
        return hits[-1] if hits else None

    def entries(self, round: int | None = None) -> list:
        if round is None:
            return list(self.data["entries"])
        return [e for e in self.data["entries"] if e["round"] == round]

    def current_entries(self) -> list:
        """Latest entry per witness (highest round) — what the arbiter reconciles. Insertion
        order = round-0 append order (stable), since later rounds overwrite by witness name."""
        latest = {}
        for e in self.data["entries"]:
            latest[e["witness"]] = e
        return list(latest.values())

    def arbiter_consensus(self, round: int | None = None):
        rounds = self.data["arbiter_rounds"]
        if round is None:
            return rounds[-1]["consensus"] if rounds else None
        for ar in rounds:
            if ar["round"] == round:
                return ar["consensus"]
        return None

    def read_opinions(self, exclude: str | None = None, round: int | None = None) -> list:
        """Compact peer view an agent reads before forming its opinion (no evidence,
        just summaries + weights) — bounded token cost, like the arbiter brief."""
        out = []
        for e in self.entries(round):
            if e["witness"] == exclude:
                continue
            out.append({"witness": e["witness"], "independence": e["independence"],
                        "grounding": e["grounding"],
                        "confidence": e["opinion"].get("confidence"),
                        "reliability_weight": e["opinion"].get("reliability_weight"),
                        "summary": e["opinion"].get("summary")})
        return out

    def set_arbiter(self, consensus: dict, round: int = 0) -> None:
        self.data["arbiter_rounds"].append({"round": round, "consensus": consensus})

    def evidence_hash(self) -> str:
        # hash the UNIQUE grounded evidence per witness (latest entry), so the hash means
        # "the grounded inputs were identical" and is INVARIANT to the number of feedback
        # rounds (re-appending a witness with revised opinion never changes its evidence).
        # In single-pass this is identical to hashing all entries (one per witness).
        blob = json.dumps([e["evidence"] for e in self.current_entries()],
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def finalize(self, stop_reason: str = "single_pass") -> None:
        self.data["stop_reason"] = stop_reason
        self.data["deterministic_evidence_hash"] = self.evidence_hash()

    def persist(self, ctx, name: str = "ledger.json") -> str:
        return ctx.save_json(self.data, name)
