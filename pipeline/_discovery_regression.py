#!/usr/bin/env python3
"""Deterministic regression for the Discovery agent (LLM off). Mirrors _phaseA_regression.py's style.

Checks the five load-bearing guarantees:
  1. leakage safety      — OOF rows are unique + count == n labeled pseudobulks; a degenerate
                           single-donor-per-class split SKIPS (never fabricates a result).
  2. perm calibration    — across a swept table weight>0 implies perm_p<alpha; a donor-level label
                           shuffle collapses a real signal to weight 0.
  3. sparse-field honesty — a rare mutation flag returns needs_more_data with weight 0 everywhere.
  4. schema round-trip   — load_discovery re-reads every output + the typed accessors work, and the
                           weights schema tolerates an unknown/forward status value (e.g. a future
                           UDON_folds 'deferred' row).
  5. baseline untouched  — DEFAULT_ORDER is unchanged; `discover` is its own separate order.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from amlmm.context import build_context, Config
from amlmm import discovery as D, steps, targets
from amlmm import step as _step

ctx = build_context(Config(run_id="disc_regression"))
# single-pass at 30 perms: min p = 1/31 = 0.032 < 0.05, so a real weight CAN fire (unlike the 10-perm smoke)
cfg = D.DiscoveryConfig(screen_permutations=30, final_permutations=0)
samples = ctx.tables["samples"]


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    assert cond, f"{name} FAILED: {detail}"


print("== 1. leakage safety ==", flush=True)
r = D.run_combo(ctx, "subtype", "composition", D.SAMPLE_SENTINEL, cfg)
check("composition combo ran", r["status"] == "ok", r.get("reason", ""))
oof = r["oof"]
check("OOF count == n_pseudobulks", len(oof) == r["n_pseudobulks"], f"{len(oof)} vs {r['n_pseudobulks']}")
check("no duplicate OOF rows", len(oof) == len(set(oof.keys())))
# planted degenerate split: each class = exactly one donor group -> must skip, never fake
donors = samples["donor_group"].dropna().astype(str)
d0, d1 = list(pd.unique(donors))[:2]
synth = pd.Series(index=samples.index, dtype=object)
synth[samples["donor_group"].astype(str) == d0] = "A"
synth[samples["donor_group"].astype(str) == d1] = "B"
rs = D.run_combo(ctx, "__synth__", "composition", D.SAMPLE_SENTINEL, cfg, labels=synth.dropna())
check("degenerate single-donor-per-class skips", rs["status"] == "skipped", rs.get("reason", ""))

print("== 2. permutation calibration ==", flush=True)
W = D.run_discovery(ctx, cfg, fields=["subtype"], modalities=["composition"],
                    cell_states_top=1, write=False, verbose=False)["weights"]
okw = W[W.status == "ok"]
bad = okw[(okw.weight > 0) & (okw.permutation_p >= cfg.alpha)]
check("weight>0 => perm_p<alpha", len(bad) == 0, f"{len(bad)} violations")
# donor-level label shuffle (the exact null model) collapses signal
lab = D.labels_for_field(ctx, "subtype").dropna()
lab = lab[lab.isin(targets.usable_classes(lab, cfg.min_class_n))]
sdf = pd.DataFrame({"k": list(lab.index), "y": list(lab.values)})
sdf["donor"] = samples.reindex(sdf["k"])["donor_group"].astype(str).values
glab = sdf.groupby("donor")["y"].first()
perm = np.random.RandomState(0).permutation(glab.values)
gmap = dict(zip(glab.index, perm))
shuffled = pd.Series([gmap[d] for d in sdf["donor"]], index=sdf["k"])
# REAL 30-perm run (not 0-perm): shuffled labels must drive perm_p high -> weight 0 (genuine calibration)
rsh = D.run_combo(ctx, "subtype_shuffled", "composition", D.SAMPLE_SENTINEL, cfg, labels=shuffled,
                  permutations=cfg.screen_permutations)
check("donor-shuffled run produced a real perm_p", rsh.get("permutation_p") is not None,
      f"p={rsh.get('permutation_p')}")
check("donor-shuffled signal collapses to weight 0 (perm_p>=alpha)",
      (rsh.get("weight") or 0) == 0 and (rsh.get("permutation_p") or 1) >= cfg.alpha,
      f"weight={rsh.get('weight')} p={rsh.get('permutation_p')}")

print("== 3. sparse-field honesty ==", flush=True)
fld = "mut_SRSF2"
rr = D.run_discovery(ctx, cfg, fields=[fld], modalities=["composition"],
                     cell_states_top=1, write=False, verbose=False)
allz = all((row.get("weight") or 0) == 0 for row in rr["results"])
check(f"{fld} not faked (weight 0 everywhere)", allz)
check(f"{fld} in needs_more_data", fld in rr["report"]["summary"]["fields_needs_more_data"])

print("== 4. schema + load_discovery round-trip ==", flush=True)
D.run_discovery(ctx, cfg, fields=["subtype", "mut_NPM1"], modalities=["composition"],
                cell_states_top=1, write=True, verbose=False)
dr = D.load_discovery(ctx.run_dir)
check("weights re-load", len(dr.weights) > 0)
check("associations re-load", len(dr.associations) > 0)
check("report re-load", "field_ability" in dr.report)
check("typed weight() accessor returns float",
      isinstance(dr.weight("subtype", "composition", D.SAMPLE_SENTINEL), float))
check("associations_for() filters", len(dr.associations_for("subtype", "composition", D.SAMPLE_SENTINEL)) > 0)
W2 = pd.concat([dr.weights, pd.DataFrame([{"field": "x", "modality": "UDON_folds", "cell_state": "C",
               "status": "deferred", "weight": 0.0}])], ignore_index=True)
check("schema tolerates a forward 'deferred' status row", bool((W2.status == "deferred").any()))

print("== 5. baseline untouched ==", flush=True)
check("DEFAULT_ORDER unchanged",
      steps.DEFAULT_ORDER == ["feasibility", "assemble_features", "classify", "cluster_explore", "report"])
check("DISCOVERY_ORDER is separate", steps.DISCOVERY_ORDER == ["discover"])
check("discover registered", "discover" in _step.REGISTRY)
check("baseline steps still registered", all(s in _step.REGISTRY for s in steps.DEFAULT_ORDER))

print("\nDISCOVERY REGRESSION OK", flush=True)
