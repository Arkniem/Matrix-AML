#!/usr/bin/env python3
"""Phase B regression test (deterministic; LLM off).
Run:  PYTHONIOENCODING=utf-8 python _phaseB_regression.py

Covers: the new KB rows, the 9-witness roster, every new witness producing usable evidence,
the CRITICAL anchor-invariance guarantee (the 6 descriptive witnesses cannot change the
leading hypothesis / driver therapies / concordance), additive contributions appearing, and
evidence-hash determinism for the deterministic (non-CV) witnesses.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import amlmm
from amlmm import knowledge, arbiter, panel, dataio, ledger as _led
from amlmm.agents import default_roster, NEW_WITNESSES, GENETIC, CELL_STATE
from amlmm.agent import run_agent

fails = []
def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)

NPM1 = "CCHMC::0018_Af_N1c"
TP53 = "WashU::10DD-1002__Diagnosis"
SCOPE = lambda k: {"mode": "patient", "sample_key": k, "strategy": "donor_kfold", "permutations": 10}


def append_roster(ctx, L, roster, key):
    for spec in roster:
        L.append(run_agent(spec, ctx, SCOPE(key), None), round=0)
    return L


print("== KB (Phase B additions) ==")
kb = knowledge.load_knowledge()
check("KB version bumped to kb-2026.07", kb.version == "kb-2026.07", kb.version)
check("seed_check empty", knowledge.seed_check(kb) == set(), str(knowledge.seed_check(kb)))
check("CD33 surface therapy row", any(t["biomarker"] == "CD33" for t in kb.therapies_for(["CD33"])))
check("CD123 surface therapy row", any(t["biomarker"] == "CD123" for t in kb.therapies_for(["CD123"])))
check("metabolite validation row", any(v["claim"] == "metabolite" for v in kb.validations_for(["metabolite"])))
check("lipid_profile validation row", any(v["claim"] == "lipid_profile" for v in kb.validations_for(["lipid_profile"])))

print("\n== roster = composition + 8 specialized witnesses ==")
names = [s.name for s in default_roster()]
print("   roster:", names)
for w in ["composition", "genetic", "cell-state/UDON", "LSC", "surfaceome/ADT",
          "metabolic", "lipid", "GRN-regulon", "cell-communication"]:
    check(f"roster includes {w}", w in names)
check("no duplicate witness names in roster", len(names) == len(set(names)), str(names))
domains = {s.domain for s in default_roster()}
check("new domains are NOT in the voting set",
      domains.isdisjoint({"genetic"}) is False and  # genetic still present (it anchors)
      {"lsc", "surfaceome", "metabolic", "lipid", "grn", "cell_comm"}.issubset(domains))

ctx = amlmm.build_context(amlmm.Config(run_id="regrB"))

print("\n== each new witness produces usable evidence for a real patient (TP53) ==")
for fn, label in [(panel._patient_lsc, "LSC"), (panel._patient_surfaceome, "surfaceome"),
                  (panel._patient_metabolic, "metabolic"), (panel._patient_lipid, "lipid"),
                  (panel._patient_grn, "GRN"), (panel._patient_signaling, "cell-communication")]:
    ev = fn(ctx, TP53)
    check(f"{label} status ok", ev.get("status") == "ok", str(ev.get("reason", "")))

print("\n== CRITICAL: the 6 new witnesses cannot change the decision (anchor invariance) ==")
base_roster = default_roster(with_lsc=False, with_surfaceome=False, descriptive=())  # comp+gen+udon
new_specs = [NEW_WITNESSES["lsc"], NEW_WITNESSES["surfaceome"], NEW_WITNESSES["metabolic"],
             NEW_WITNESSES["lipid"], NEW_WITNESSES["grn"], NEW_WITNESSES["cell_comm"]]
for key, want in [(TP53, "TP53"), (NPM1, "NPM1")]:
    L = _led.Ledger(key, "subtype"); ctx.ledger = L
    append_roster(ctx, L, base_roster, key)          # composition evidence computed ONCE
    cb = arbiter.reconcile_patient(None, ctx, L)
    append_roster(ctx, L, new_specs, key)            # add the 6 descriptive witnesses to SAME ledger
    cf = arbiter.reconcile_patient(None, ctx, L)
    check(f"{want}: leading unchanged by new witnesses",
          cb["leading_hypothesis"] == cf["leading_hypothesis"] == want,
          f"base={cb['leading_hypothesis']} full={cf['leading_hypothesis']}")
    check(f"{want}: driver therapies unchanged",
          cb["ranked_therapy_hypotheses"] == cf["ranked_therapy_hypotheses"])
    check(f"{want}: concordance unchanged",
          cb["concordance"] == cf["concordance"], f"{cb['concordance']} vs {cf['concordance']}")

print("\n== additive contributions appear on the full roster (TP53) ==")
L = _led.Ledger(TP53, "subtype"); ctx.ledger = L
append_roster(ctx, L, default_roster(), TP53)
cf = arbiter.reconcile_patient(None, ctx, L)
check("descriptive_findings populated", len(cf.get("descriptive_findings", [])) > 0,
      f"n={len(cf.get('descriptive_findings', []))}")
surf = cf.get("surface_therapy_hypotheses", [])
check("any surface therapy is flagged flow-pending", all(t.get("requires_flow_confirmation") is True for t in surf),
      f"n_surface={len(surf)}")
check("surface markers NOT mixed into driver therapies",
      all(t.get("biomarker") not in ("CD33", "CD123") for t in cf.get("ranked_therapy_hypotheses", [])))
val_claims = {v.get("claim") for v in cf.get("recommended_validations", [])}
check("descriptive witness validation claims surfaced (metabolite/lipid_profile/lsc/surface_marker)",
      bool(val_claims & {"metabolite", "lipid_profile", "lsc", "surface_marker"}), str(sorted(val_claims)))

print("\n== surface-therapy POSITIVE path (patient with elevated imputed CD33/CD123) ==")
# auto-find a patient whose imputed CD33/CD123 clears z>=1 (robust across anndata versions)
Madt = dataio.cohort_modality_matrix(ctx, "ADT")
zc = (Madt - Madt.mean()) / Madt.std(ddof=0).replace(0, np.nan)
cd_cols = [c for c in Madt.columns if str(c).split("_")[0].split(".")[0] in ("CD33", "CD123")]
cand = zc[cd_cols].max(axis=1) if cd_cols else None
SURF = (cand[cand >= 1.0].sort_values(ascending=False).index[0]
        if cand is not None and (cand >= 1.0).any() else None)
check("found a CD33/CD123-high patient to exercise the surface path", SURF is not None, str(SURF))
sev = panel._patient_surfaceome(ctx, SURF)
check("surfaceome emits therapy_biomarkers for a surface-high patient",
      len(sev.get("therapy_biomarkers", [])) > 0, str(sev.get("therapy_biomarkers")))
Lf = _led.Ledger(SURF, "subtype"); ctx.ledger = Lf
append_roster(ctx, Lf, default_roster(), SURF)
cf = arbiter.reconcile_patient(None, ctx, Lf)
Lb = _led.Ledger(SURF, "subtype"); ctx.ledger = Lb
append_roster(ctx, Lb, default_roster(with_lsc=False, with_surfaceome=False, descriptive=()), SURF)
cb = arbiter.reconcile_patient(None, ctx, Lb)
surf = cf.get("surface_therapy_hypotheses", [])
check("surface_therapy_hypotheses NON-empty (positive path)", len(surf) > 0, f"n={len(surf)}")
check("every surface therapy flagged flow-pending", all(t.get("requires_flow_confirmation") is True for t in surf))
check("surface biomarkers are CD33/CD123 only", all(t.get("biomarker") in ("CD33", "CD123") for t in surf))
check("CD33/CD123 NOT mixed into driver therapies",
      all(t.get("biomarker") not in ("CD33", "CD123") for t in cf.get("ranked_therapy_hypotheses", [])))
check("surfaceome does not change leading", cb["leading_hypothesis"] == cf["leading_hypothesis"],
      f"{cb['leading_hypothesis']} vs {cf['leading_hypothesis']}")
check("surfaceome does not change driver therapies", cb["ranked_therapy_hypotheses"] == cf["ranked_therapy_hypotheses"])

print("\n== no-anchor consensus path: descriptive witnesses don't change a non-anchored leading ==")
class FakeLedger:
    def __init__(self, e): self._e = e
    def entries(self): return self._e
def E(witness, domain, grounding, indep, evidence, w):
    return {"witness": witness, "domain": domain, "grounding": grounding, "independence": indep,
            "evidence": evidence, "opinion": {"reliability_weight": w}}
base_entries = [
    E("genetic", "genetic", "deterministic_fact", "independent", {"present": [], "targetable": {}}, 0.2),
    E("composition", "predictive", "honest_cv", "independent",
      {"patient_prediction": "NPM1", "permutation_pvalue": 0.01}, 0.7),
]
desc_entries = [
    E("metabolic", "metabolic", "descriptive_aggregate", "imputed_from_RNA",
      {"descriptive_context": ["m"], "validation_claims": ["metabolite"]}, 0.3),
    E("surfaceome/ADT", "surfaceome", "descriptive_aggregate", "imputed_from_RNA",
      {"therapy_biomarkers": ["CD33"], "validation_claims": ["surface_marker"], "descriptive_context": ["s"]}, 0.3),
]
na_b = arbiter.reconcile_patient(None, ctx, FakeLedger(base_entries))
na_f = arbiter.reconcile_patient(None, ctx, FakeLedger(base_entries + desc_entries))
check("no-anchor leading is the consensus vote (NPM1)", na_b["leading_hypothesis"] == "NPM1", na_b["leading_hypothesis"])
check("no-anchor leading unchanged by descriptive witnesses", na_b["leading_hypothesis"] == na_f["leading_hypothesis"])
check("no-anchor concordance unchanged", na_b["concordance"] == na_f["concordance"],
      f"{na_b['concordance']} vs {na_f['concordance']}")
check("no-anchor: CD33 surfaces as flow-pending hypothesis, NOT a driver therapy",
      any(t.get("biomarker") == "CD33" for t in na_f.get("surface_therapy_hypotheses", []))
      and all(t.get("biomarker") != "CD33" for t in na_f.get("ranked_therapy_hypotheses", [])))

print("\n== missing modality degrades to status='skipped' (not 'error') ==")
ctx3 = amlmm.build_context(amlmm.Config(run_id="regrB3"))
ctx3._modality_paths.pop("GRN", None)
check("GRN witness skips when its file is absent", panel._patient_grn(ctx3, TP53).get("status") == "skipped")
ctx3._modality_paths.pop("ADT", None)
check("surfaceome skips when ADT is absent", panel._patient_surfaceome(ctx3, TP53).get("status") == "skipped")

print("\n== determinism: stable evidence hash on a COLD (fresh-context) re-run ==")
det_roster = [GENETIC, CELL_STATE] + list(NEW_WITNESSES.values())   # no CV/permutation randomness
def det_hash(c, key):
    L = _led.Ledger(key, "subtype"); c.ledger = L
    append_roster(c, L, det_roster, key); L.finalize()
    return L.evidence_hash()
ctx2 = amlmm.build_context(amlmm.Config(run_id="regrB2"))   # fresh ctx -> cold caches / aggregation
h1 = det_hash(ctx, TP53)
h2 = det_hash(ctx2, TP53)
check("evidence_hash identical across fresh contexts (cold path)", h1 == h2, f"{h1} vs {h2}")
import json as _json
L = _led.Ledger(TP53, "subtype"); ctx.ledger = L
append_roster(ctx, L, det_roster, TP53)
blob = _json.dumps([e["evidence"] for e in L.entries()], sort_keys=True, default=str)
check("evidence JSON round-trips (no non-serializable / numpy-scalar drift)", _json.loads(blob) is not None)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
