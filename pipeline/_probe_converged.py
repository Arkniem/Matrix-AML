#!/usr/bin/env python3
"""Probe: which break path does each Phase C scenario take?
We instrument feedback._converged and the no-revision path by re-implementing the
loop body's accounting via wrappers, counting how often each branch fires."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amlmm
from amlmm import arbiter, panel, feedback, ledger as _led
from amlmm.agent import AgentResult

# ---- instrument _converged to record every call + its boolean result ----
calls = {"converged_calls": [], "converged_true": 0}
_orig_converged = feedback._converged
def spy_converged(cur, prev):
    res = _orig_converged(cur, prev)
    calls["converged_calls"].append(res)
    if res:
        calls["converged_true"] += 1
    return res
feedback._converged = spy_converged

# ---- instrument deliberate to record how the loop exits ----
# We can't easily see the local stop path, so re-run the loop logic by tracking
# rounds_run + stop_reason + whether _converged ever returned True.

def reset():
    calls["converged_calls"] = []
    calls["converged_true"] = 0

NPM1 = "CCHMC::0018_Af_N1c"
TP53 = "WashU::10DD-1002__Diagnosis"

def mk_ledger(specs):
    L = _led.Ledger("fake", "subtype")
    for name, dom, gr, ind, ev, op in specs:
        L.append(AgentResult(name=name, domain=dom, grounding=gr, independence=ind,
                             evidence=ev, opinion=op), round=0)
    return L

ctx = amlmm.build_context(amlmm.Config(run_id="probeC"))

print("=== SAFETY scenarios (real patients) ===")
for key, want in [(TP53, "TP53"), (NPM1, "NPM1")]:
    reset()
    rep = panel.run_patient_panel(ctx, key, permutations=10, client=None, max_rounds=2)
    dl = rep["deliberation"]
    print(f"{want}: stop_reason={dl['stop_reason']} rounds_run={dl['rounds_run']} "
          f"_converged calls={calls['converged_calls']} converged_true={calls['converged_true']} "
          f"baseline_conc={dl['drift']['baseline_concordance']} final_conc={dl['drift']['final_concordance']}")

print("\n=== deterministic conflict_ledger scenario ===")
def conflict_ledger():
    return mk_ledger([
        ("genetic", "genetic", "deterministic_fact", "independent",
         {"present": ["TP53"], "targetable": {}}, {"reliability_weight": 0.85}),
        ("composition", "predictive", "honest_cv", "independent",
         {"patient_prediction": "FLT3", "permutation_pvalue": 0.01}, {"reliability_weight": 0.8}),
    ])
reset()
ctx.ledger = conflict_ledger()
c0 = arbiter.reconcile_patient(None, ctx, ctx.ledger); ctx.ledger.set_arbiter(c0, round=0)
r = feedback.deliberate(ctx, ctx.ledger, None, max_rounds=2)
print(f"conflict_ledger: stop_reason={r['stop_reason']} rounds_run={r['rounds_run']} "
      f"_converged calls={calls['converged_calls']} converged_true={calls['converged_true']} "
      f"baseline_conc={r['baseline']['concordance']} final_conc={r['consensus']['concordance']}")

print("\n=== no-anchor scenario ===")
reset()
Lna = mk_ledger([
    ("genetic", "genetic", "deterministic_fact", "independent", {"present": [], "targetable": {}}, {"reliability_weight": 0.2}),
    ("composition", "predictive", "honest_cv", "independent",
     {"patient_prediction": "NPM1", "permutation_pvalue": 0.01}, {"reliability_weight": 0.8}),
    ("cell-state/UDON", "cell_state", "discovery", "rna_derived",
     {"active_programs": [{"marks": "Inv16"}]}, {"reliability_weight": 0.4}),
])
ctx.ledger = Lna
c0 = arbiter.reconcile_patient(None, ctx, Lna); Lna.set_arbiter(c0, round=0)
dna = feedback.deliberate(ctx, Lna, None, max_rounds=2)
print(f"no-anchor: stop_reason={dna['stop_reason']} rounds_run={dna['rounds_run']} "
      f"_converged calls={calls['converged_calls']} converged_true={calls['converged_true']}")

print("\n=== conflict_triggered (skip) scenario ===")
reset()
Lcc = mk_ledger([
    ("genetic", "genetic", "deterministic_fact", "independent", {"present": ["NPM1"], "targetable": {}}, {"reliability_weight": 0.85}),
    ("composition", "predictive", "honest_cv", "independent",
     {"patient_prediction": "NPM1", "permutation_pvalue": 0.01}, {"reliability_weight": 0.8}),
])
ctx.ledger = Lcc
c0 = arbiter.reconcile_patient(None, ctx, Lcc); Lcc.set_arbiter(c0, round=0)
dcc = feedback.deliberate(ctx, Lcc, None, max_rounds=2, mode="conflict_triggered")
print(f"conflict_triggered: stop_reason={dcc['stop_reason']} rounds_run={dcc['rounds_run']} "
      f"_converged calls={calls['converged_calls']} converged_true={calls['converged_true']}")

print("\n=== SUMMARY: did _converged() EVER return True across all scenarios? ===")
