#!/usr/bin/env python3
"""Phase C regression test (deterministic; LLM off).
Run:  PYTHONIOENCODING=utf-8 python _phaseC_regression.py

THE headline guarantee: the guarded feedback loop refines confidence but can NEVER flip an
anchored decision OR change the driver therapies. Plus: anchor dominance at adversarial vote
weight, deterministic deference, convergence (and the cap branch), evidence immutability across
rounds, no false deference without an anchor, the genetic witness never defers, descriptive
witnesses never vote, the drift watchdog fires when it should, conflict_triggered skips, the
max_rounds=0 single-pass contract, and PATIENT.md rendering.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amlmm
from amlmm import arbiter, panel, feedback, ledger as _led
from amlmm.agent import AgentResult

fails = []
def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)

NPM1 = "CCHMC::0018_Af_N1c"
TP53 = "WashU::10DD-1002__Diagnosis"
DESCRIPTIVE_DOMAINS = {"lsc", "surfaceome", "metabolic", "lipid", "grn", "cell_comm"}

def mk_ledger(specs):
    L = _led.Ledger("fake", "subtype")
    for name, dom, gr, ind, ev, op in specs:
        L.append(AgentResult(name=name, domain=dom, grounding=gr, independence=ind,
                             evidence=ev, opinion=op), round=0)
    return L

def jeq(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)

ctx = amlmm.build_context(amlmm.Config(run_id="regrC"))

print("== SAFETY: continuous deliberation never flips an anchored decision OR its therapies ==")
for key, want in [(TP53, "TP53"), (NPM1, "NPM1")]:
    rep = panel.run_patient_panel(ctx, key, permutations=10, client=None, max_rounds=2)
    c, dl = rep["consensus"], rep["deliberation"]
    d, bl = dl["drift"], dl["baseline"]
    check(f"{want}: still leads {want} after deliberation", c["leading_hypothesis"] == want, c["leading_hypothesis"])
    check(f"{want}: leading confirmed by genetics", c["leading_confirmed_by_genetics"] is True)
    check(f"{want}: leading unchanged baseline->final", d["leading_changed"] is False)
    check(f"{want}: NO groupthink warning", d["groupthink_warning"] is False)
    check(f"{want}: loop converged", dl["stop_reason"] == "converged", dl["stop_reason"])
    check(f"{want}: concordance rose or held", float(d["final_concordance"]) >= float(d["baseline_concordance"]),
          f"{d['baseline_concordance']}->{d['final_concordance']}")
    # therapies / surface / validations are keyed on immutable evidence -> invariant under deliberation
    for fld in ("ranked_therapy_hypotheses", "surface_therapy_hypotheses", "recommended_validations"):
        check(f"{want}: {fld} invariant baseline->final", jeq(bl.get(fld), c.get(fld)))
    cur = ctx.ledger.current_entries()
    gen = next(e for e in cur if e["witness"] == "genetic")
    check(f"{want}: genetic witness NEVER defers", not gen["opinion"].get("revised"))
    revised = [e["witness"] for e in cur if e["opinion"].get("revised")]
    check(f"{want}: a conflicting voting witness deferred", len(revised) >= 1, str(revised))
    desc_names = {e["witness"] for e in cur if e["domain"] in DESCRIPTIVE_DOMAINS}
    check(f"{want}: descriptive witnesses cast NO vote (absent from consistency)",
          desc_names.isdisjoint(set(c.get("per_witness_consistency", {}))), str(desc_names))
    check(f"{want}: deliberation_rounds excludes the baseline", dl.get("deliberation_rounds") >= 1,
          str(dl.get("deliberation_rounds")))

print("\n== ADVERSARIAL: a conflicting voter at weight 1e9 cannot overturn the anchor ==")
Ladv = mk_ledger([
    ("genetic", "genetic", "deterministic_fact", "independent", {"present": ["TP53"], "targetable": {}}, {"reliability_weight": 0.01}),
    ("composition", "predictive", "honest_cv", "independent", {"patient_prediction": "FLT3", "permutation_pvalue": 0.01}, {"reliability_weight": 1e9}),
    ("cell-state/UDON", "cell_state", "discovery", "rna_derived", {"active_programs": [{"marks": "NPM1"}]}, {"reliability_weight": 1e9}),
])
ctx.ledger = Ladv
c0 = arbiter.reconcile_patient(None, ctx, Ladv); Ladv.set_arbiter(c0, round=0)
adv = feedback.deliberate(ctx, Ladv, None, max_rounds=2)
check("1e9 votes cannot flip the anchor", adv["consensus"]["leading_hypothesis"] == "TP53", adv["consensus"]["leading_hypothesis"])
check("anchor stays genetically confirmed", adv["consensus"]["leading_confirmed_by_genetics"] is True)

print("\n== drift watchdog fires when (and only when) it should ==")
gw = lambda b, f: feedback._drift(b, f)["groupthink_warning"]
check("groupthink_warning True: leading changed AND not confirmed",
      gw({"leading_hypothesis": "NPM1", "concordance": 0.5, "leading_confirmed_by_genetics": False},
         {"leading_hypothesis": "FLT3", "concordance": 0.5, "leading_confirmed_by_genetics": False}) is True)
check("groupthink_warning False: leading changed BUT genetically confirmed",
      gw({"leading_hypothesis": "NPM1", "concordance": 0.5, "leading_confirmed_by_genetics": False},
         {"leading_hypothesis": "FLT3", "concordance": 0.5, "leading_confirmed_by_genetics": True}) is False)
check("groupthink_warning False: leading unchanged",
      gw({"leading_hypothesis": "TP53", "concordance": 0.5, "leading_confirmed_by_genetics": True},
         {"leading_hypothesis": "TP53", "concordance": 0.9, "leading_confirmed_by_genetics": True}) is False)

print("\n== _converged unit ==")
base = {"leading_hypothesis": "TP53", "concordance": 0.7, "per_witness_consistency": {"a": "agree"}}
check("_converged True: identical decision fields", feedback._converged(dict(base), dict(base)) is True)
check("_converged False: concordance differs",
      feedback._converged({**base, "concordance": 0.8}, base) is False)
check("_converged False: leading differs",
      feedback._converged({**base, "leading_hypothesis": "FLT3"}, base) is False)

print("\n== deterministic + cap semantics (synthetic anchored conflict) ==")
def conflict_ledger():
    return mk_ledger([
        ("genetic", "genetic", "deterministic_fact", "independent", {"present": ["TP53"], "targetable": {}}, {"reliability_weight": 0.85}),
        ("composition", "predictive", "honest_cv", "independent", {"patient_prediction": "FLT3", "permutation_pvalue": 0.01}, {"reliability_weight": 0.8}),
    ])
def run_delib(L, **kw):
    ctx.ledger = L
    c0 = arbiter.reconcile_patient(None, ctx, L); L.set_arbiter(c0, round=0)
    return feedback.deliberate(ctx, L, None, **kw)
r1, r2 = run_delib(conflict_ledger(), max_rounds=2), run_delib(conflict_ledger(), max_rounds=2)
check("deliberation leading deterministic", r1["consensus"]["leading_hypothesis"] == r2["consensus"]["leading_hypothesis"] == "TP53")
check("deliberation concordance deterministic", r1["consensus"]["concordance"] == r2["consensus"]["concordance"])
check("anchored conflict: concordance rises after deference",
      float(r1["consensus"]["concordance"]) > float(r1["baseline"]["concordance"]),
      f"{r1['baseline']['concordance']}->{r1['consensus']['concordance']}")
# max_rounds=1 boundary: revises once, exits at the cap labeled 'max_rounds'
cap = run_delib(conflict_ledger(), max_rounds=1)
check("max_rounds=1 exits at the cap (stop_reason='max_rounds')", cap["stop_reason"] == "max_rounds", cap["stop_reason"])
check("max_rounds=1 ran exactly 1 deliberation round", cap["deliberation_rounds"] == 1, str(cap["deliberation_rounds"]))
check("max_rounds=1 anchor still leads", cap["consensus"]["leading_hypothesis"] == "TP53")
# max_rounds=0 via direct call -> single_pass, zero deliberation rounds
zr = run_delib(conflict_ledger(), max_rounds=0)
check("max_rounds=0 -> single_pass, 0 deliberation rounds", zr["stop_reason"] == "single_pass" and zr["deliberation_rounds"] == 0)

print("\n== evidence IMMUTABLE across rounds: hash unchanged on the SAME ledger ==")
Lh = conflict_ledger(); ctx.ledger = Lh
c0 = arbiter.reconcile_patient(None, ctx, Lh); Lh.set_arbiter(c0, round=0)
Lh.finalize(); h_before = Lh.evidence_hash()
feedback.deliberate(ctx, Lh, None, max_rounds=2)
check("evidence_hash unchanged after deliberation (same ledger)", h_before == Lh.evidence_hash(),
      f"{h_before} vs {Lh.evidence_hash()}")

print("\n== no-anchor: deliberation does NOT meddle (no false deference) ==")
Lna = mk_ledger([
    ("genetic", "genetic", "deterministic_fact", "independent", {"present": [], "targetable": {}}, {"reliability_weight": 0.2}),
    ("composition", "predictive", "honest_cv", "independent", {"patient_prediction": "NPM1", "permutation_pvalue": 0.01}, {"reliability_weight": 0.8}),
    ("cell-state/UDON", "cell_state", "discovery", "rna_derived", {"active_programs": [{"marks": "Inv16"}]}, {"reliability_weight": 0.4}),
])
dna = run_delib(Lna, max_rounds=2)
check("no-anchor: leading is the consensus vote (NPM1)", dna["consensus"]["leading_hypothesis"] == "NPM1")
check("no-anchor: leading unchanged (no confirmed anchor -> no deference)", dna["drift"]["leading_changed"] is False)
check("no-anchor: concordance unchanged (nobody deferred)",
      dna["drift"]["baseline_concordance"] == dna["drift"]["final_concordance"])

print("\n== conflict_triggered mode skips a concordant round 0 (ledger + report agree) ==")
Lcc = mk_ledger([
    ("genetic", "genetic", "deterministic_fact", "independent", {"present": ["NPM1"], "targetable": {}}, {"reliability_weight": 0.85}),
    ("composition", "predictive", "honest_cv", "independent", {"patient_prediction": "NPM1", "permutation_pvalue": 0.01}, {"reliability_weight": 0.8}),
])
dcc = run_delib(Lcc, max_rounds=2, mode="conflict_triggered")
check("conflict_triggered skips on a concordant round 0", dcc["stop_reason"] == "no_conflict", dcc["stop_reason"])
check("ledger stop_reason MATCHES report (no two-sources-of-truth)",
      Lcc.data["stop_reason"] == dcc["stop_reason"], f"{Lcc.data['stop_reason']} vs {dcc['stop_reason']}")
check("conflict_triggered skip reports 0 deliberation rounds", dcc["deliberation_rounds"] == 0)

print("\n== max_rounds=0 single-pass contract + PATIENT.md rendering ==")
panel.run_patient_panel(ctx, NPM1, permutations=10, client=None, max_rounds=0)
rep0_led = ctx.ledger
h_single = rep0_led.data["deterministic_evidence_hash"]
md0 = open(ctx.path("PATIENT.md"), encoding="utf-8").read()
check("single-pass: stop_reason 'single_pass'", rep0_led.data["stop_reason"] == "single_pass")
check("single-pass PATIENT.md has NO Deliberation section", "### Deliberation" not in md0)
panel.run_patient_panel(ctx, NPM1, permutations=10, client=None, max_rounds=2)
h_multi = ctx.ledger.data["deterministic_evidence_hash"]
md2 = open(ctx.path("PATIENT.md"), encoding="utf-8").read()
check("cross-run evidence_hash identical single-pass vs 2-round", h_single == h_multi, f"{h_single} vs {h_multi}")
check("multi-round PATIENT.md HAS a Deliberation section", "### Deliberation" in md2 and "Deliberation rounds:" in md2)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
