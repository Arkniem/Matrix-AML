#!/usr/bin/env python3
"""Stage-3 validation: full run_discovery iteration -> the three tables + index + report.
Mirrors the plan's smoke spec (subtype + a mutation flag x composition+ADT, top-3 states, screen-only)
and asserts: every output writes + re-parses; per-combo association rows == n_pseudobulks (leakage-clean
OOF, none lost/dup); markers are held-out-validated; feature_space stored only where Deploy would query;
the mutation-predictability matrix + optimize-me list are populated."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from amlmm.context import build_context, Config
from amlmm import discovery as D

ctx = build_context(Config(run_id="disc_stage3_smoke"))
cfg = D.DiscoveryConfig(screen_permutations=10, final_permutations=0)   # single-pass, fast
res = D.run_discovery(ctx, cfg, fields=["subtype", "mut_NPM1"],
                      modalities=["composition", "ADT"], cell_states_top=3, verbose=True)

rd = ctx.run_dir
print("\n==== files ====")
for name in ("discovery_weights.tsv", "discovery_markers.tsv", "discovery_associations.tsv",
             "discovery_associations_index.json", "discovery_report.json", "DISCOVERY.md"):
    p = os.path.join(rd, name)
    assert os.path.exists(p), f"MISSING output: {name}"
    print(f"  {name}: {os.path.getsize(p)} bytes")

# --- weights table re-parses; one row per attempted combo ---
W = pd.read_csv(os.path.join(rd, "discovery_weights.tsv"), sep="\t")
assert len(W) == len(res["results"]), f"weights rows {len(W)} != results {len(res['results'])}"
for c in ("field", "modality", "cell_state", "status", "weight", "permutation_p", "fidelity"):
    assert c in W.columns, f"weights missing col {c}"
print(f"\nweights: {len(W)} rows; status counts = {W['status'].value_counts().to_dict()}")

# --- associations: per ok-combo, #OOF rows == n_pseudobulks (no leakage rows lost/dup) ---
A = pd.read_csv(os.path.join(rd, "discovery_associations.tsv"), sep="\t")
okres = [r for r in res["results"] if r.get("status") == "ok"]
for r in okres:
    sub = A[(A.field == r["field"]) & (A.modality == r["modality"]) & (A.cell_state == r["cell_state"])]
    assert len(sub) == r["n_pseudobulks"], \
        f"assoc {len(sub)} != n_pb {r['n_pseudobulks']} for {r['field']}x{r['modality']}x{r['cell_state']}"
    assert sub["pseudobulk_id"].is_unique, "duplicate pseudobulk in associations"
print(f"associations: {len(A)} OOF rows across {len(okres)} ok combos (counts == n_pseudobulks ✓)")

# --- association index: keys == ok combos; feature_space iff weight>0 or gate_accept ---
with open(os.path.join(rd, "discovery_associations_index.json"), encoding="utf-8") as fh:
    IDX = json.load(fh)
assert len(IDX) == len(okres), f"index {len(IDX)} != ok combos {len(okres)}"
for r in okres:
    key = f"{r['field']}|{r['modality']}|{r['cell_state']}"
    assert key in IDX, f"index missing {key}"
    has_fs = "feature_space" in IDX[key]
    want_fs = (r.get("weight") or 0) > 0 or r.get("gate_accept")
    assert has_fs == bool(want_fs), f"feature_space presence wrong for {key} (w={r.get('weight')})"
print(f"index: {len(IDX)} combos; feature_space gating ✓")

# --- markers: only held-out-validated, every row has a feature ---
M = pd.read_csv(os.path.join(rd, "discovery_markers.tsv"), sep="\t") if \
    os.path.getsize(os.path.join(rd, "discovery_markers.tsv")) > 0 else pd.DataFrame()
if len(M):
    assert M["feature"].notna().all(), "marker row with no feature"
    assert M["heldout_auroc"].fillna(0).ge(0.6).all() or M["heldout_auroc"].isna().any() is False
print(f"markers: {len(M)} held-out-validated rows")

# --- report: structure + headline matrix + optimize-me ---
with open(os.path.join(rd, "discovery_report.json"), encoding="utf-8") as fh:
    REP = json.load(fh)
for k in ("summary", "field_ability", "mutation_predictability", "optimize_me", "config"):
    assert k in REP, f"report missing {k}"
assert "subtype" in REP["field_ability"], "subtype missing from field_ability"
assert "mut_NPM1" in REP["mutation_predictability"], "mut_NPM1 missing from mutation matrix"
md = open(os.path.join(rd, "DISCOVERY.md"), encoding="utf-8").read()
assert "Field predictability" in md and len(md) > 200, "DISCOVERY.md malformed"
print(f"report: summary={REP['summary']}")
print(f"  mut_NPM1 predictability: {REP['mutation_predictability'].get('mut_NPM1')}")
print(f"  optimize_me ({len(REP['optimize_me'])}): {[o['field']+':'+o['issue'][:40] for o in REP['optimize_me'][:4]]}")

print("\nSTAGE 3 TEST OK")
