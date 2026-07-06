#!/usr/bin/env python3
"""Phase A post-review regression test (deterministic; no LLM needed).

Covers the bugs the adversarial review confirmed + the no-regression cases.
Run:  PYTHONIOENCODING=utf-8 python _phaseA_regression.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amlmm
from amlmm import knowledge, arbiter, panel

fails = []
def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


class FakeLedger:
    def __init__(self, entries): self._e = entries
    def entries(self): return self._e


def entry(witness, domain, grounding, indep, evidence, w):
    return {"witness": witness, "domain": domain, "grounding": grounding,
            "independence": indep, "evidence": evidence,
            "opinion": {"reliability_weight": w}}


print("== KB load + schema validation + seed_check ==")
kb = knowledge.load_knowledge()
check("KB version present", kb.version.startswith("kb-"), kb.version)
check("seed_check empty (TARGETABLE covered)", knowledge.seed_check(kb) == set(),
      str(knowledge.seed_check(kb)))
check("APL therapy row present", any(t["biomarker"] == "t15_17" for t in kb.therapies_for(["t15_17"])))
check("KMT2Ar therapy row present", any(t["biomarker"] == "kmt2a" for t in kb.therapies_for(["kmt2a"])))
check("APL validation row present", any(v["claim"] == "t15_17" for v in kb.validations_for(["t15_17"])))

# header-validation: a misspelled header must raise, not KeyError later
import tempfile, shutil
tmp = tempfile.mkdtemp()
shutil.copy(os.path.join(knowledge.KB_DIR, "VERSION"), os.path.join(tmp, "VERSION"))
with open(os.path.join(tmp, "validation_rules.tsv"), "w") as f:
    f.write("claim_type\tvalidation\tsource\nmutation\tNGS\tstd\n")
with open(os.path.join(tmp, "biomarker_drug.tsv"), "w") as f:
    f.write("gene\tdrug\tevidence_level\tsource\nFLT3\tgilt\tguideline\tNCCN\n")  # bad header 'gene'
raised = False
try:
    knowledge.load_knowledge(tmp)
except ValueError as e:
    raised = "biomarker" in str(e)
check("bad TSV header raises clear ValueError (not latent KeyError)", raised)

print("\n== arbiter: APL/t(15;17) anchors and is NOT outranked by an imputed FLT3 ==")
led = FakeLedger([
    entry("genetic", "genetic", "deterministic_fact", "independent",
          {"present": ["t15_17"], "targetable": {}}, 0.85),
    entry("composition", "predictive", "honest_cv", "independent",
          {"patient_prediction": "FLT3", "permutation_pvalue": 0.01}, 0.7),
])
class C: pass
ctx = C(); ctx.knowledge = kb
con = arbiter.reconcile_patient(None, ctx, led)
check("APL leads (not FLT3)", con["leading_hypothesis"] == "APL", con["leading_hypothesis"])
check("APL confirmed by genetics", con["leading_confirmed_by_genetics"] is True)
check("ATRA/ATO therapy emitted", any("ATRA" in t["drug"] for t in con["ranked_therapy_hypotheses"]),
      str([t["drug"][:25] for t in con["ranked_therapy_hypotheses"]]))
check("urgent PML-RARA validation emitted",
      any("PML-RARA" in v["validation"] for v in con["recommended_validations"]))
check("FLT3 recorded as a conflict", "composition" in str(con["conflicts"]), con["conflicts"])

print("\n== arbiter: unanchored 'multi' generic token never becomes the lead ==")
led2 = FakeLedger([
    entry("genetic", "genetic", "deterministic_fact", "independent", {"present": [], "targetable": {}}, 0.2),
    entry("composition", "predictive", "honest_cv", "independent",
          {"patient_prediction": "multi", "permutation_pvalue": 0.2}, 0.6),
    entry("cell-state/UDON", "cell_state", "discovery", "rna_derived",
          {"active_programs": [{"marks": "NPM1"}]}, 0.5),
])
con2 = arbiter.reconcile_patient(None, ctx, led2)
check("leading is not the generic 'multi'", con2["leading_hypothesis"] != "multi", str(con2["leading_hypothesis"]))

print("\n== CRITICAL: two patients on ONE shared ctx both keep the genetic anchor ==")
NPM1 = "CCHMC::0018_Af_N1c"
TP53 = "WashU::10DD-1002__Diagnosis"
real = amlmm.build_context(amlmm.Config(run_id="regr_shared"))
g1 = panel._patient_genetic(real, NPM1)        # first build + cache
g2 = panel._patient_genetic(real, TP53)        # <-- pre-fix this raised ValueError (swallowed)
check("patient #1 genetic ok + present", g1.get("status") == "ok" and len(g1.get("present", [])) > 0,
      str(g1.get("present")))
check("patient #2 genetic ok + present (no DataFrame-truthiness crash)",
      g2.get("status") == "ok" and len(g2.get("present", [])) > 0, str(g2.get("present")))
check("NPM1 patient has NPM1 driver", "NPM1" in g1.get("present", []), str(g1.get("present")))
check("TP53 patient has TP53 driver", "TP53" in g2.get("present", []), str(g2.get("present")))

print("\n== no-regression: NPM1 and TP53 still anchor correctly (full deterministic panel, LLM off) ==")
for key, want in [(NPM1, "NPM1"), (TP53, "TP53")]:
    ctxp = amlmm.build_context(amlmm.Config(run_id="regr_" + want))
    led3 = type("L", (), {})()
    # build the ledger via the real roster gather, but skip the LLM by passing client=None-ish:
    from amlmm import ledger as _led
    from amlmm.agents import default_roster
    from amlmm.agent import run_agent
    L = _led.Ledger(key, "subtype")
    ctxp.ledger = L
    for spec in default_roster(["composition"], True, True):
        res = run_agent(spec, ctxp, {"mode": "patient", "sample_key": key,
                                     "strategy": "donor_kfold", "permutations": 10}, None)
        L.append(res, round=0)
    con3 = arbiter.reconcile_patient(None, ctxp, L)
    check(f"{want} case leads {want}", con3["leading_hypothesis"] == want,
          f"got {con3['leading_hypothesis']}, concordance {con3['concordance']}, "
          f"confirmed {con3['leading_confirmed_by_genetics']}")

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
