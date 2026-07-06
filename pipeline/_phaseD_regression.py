#!/usr/bin/env python3
"""Phase D regression test — retrospective-validation harness correctness.
Run:  PYTHONIOENCODING=utf-8 python _phaseD_regression.py

Analytical correctness (this harness has no decision-path invariants): the stats helpers
behave on hand-checkable inputs, the label binarizations are right, cohort coverage matches
the data, and the report + VALIDATION.md render with honest n / underpowered flags.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import amlmm
from amlmm import retrospective as R

fails = []
def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)

print("== label binarization ==")
check("_norm_eln Favorable", R._norm_eln("Favorable") == "Favorable")
check("_norm_eln 'Adverse risk' -> Adverse", R._norm_eln("Adverse risk") == "Adverse")
check("_norm_eln Intermediate", R._norm_eln("Intermediate") == "Intermediate")
check("_norm_eln blank/unknown -> None", R._norm_eln("") is None and R._norm_eln("Unknown") is None)
check("_responder '(responder)' -> True", R._responder("CR (responder)") is True)
check("_responder 'Adverse' -> False", R._responder("Adverse") is False)
check("_responder ambiguous 'Favorable' -> None", R._responder("Favorable") is None)

print("\n== _fisher2x2 on hand-checkable inputs ==")
# perfect positive association
a = pd.Series([True]*10 + [False]*10)
b = pd.Series([True]*10 + [False]*10)
r = R._fisher2x2(a, b)
check("perfect association: n=20, table diagonal", r["n"] == 20 and r["table"] == [[10, 0], [0, 10]])
check("perfect association: small p", r["p"] is not None and r["p"] < 0.001, str(r["p"]))
check("perfect association: underpowered flag at n=20 is False (>=20)", r["underpowered"] is False)
# no association (independent)
import numpy as np
rng = np.random.default_rng(0)
x = pd.Series(rng.integers(0, 2, 60).astype(bool))
y = pd.Series(rng.integers(0, 2, 60).astype(bool))
r2 = R._fisher2x2(x, y)
check("independent: p not significant", r2["p"] is not None and r2["p"] > 0.05, str(r2["p"]))
# all-null overlap -> n=0
r3 = R._fisher2x2(pd.Series([None, None]), pd.Series([None, None]))
check("no overlap -> n=0, underpowered", r3["n"] == 0 and r3["underpowered"] is True)

print("\n== _assoc uses the explicit both-non-null support (env-independent) ==")
tt = pd.DataFrame({
    "g": ["Adverse", "Adverse", "Favorable", None, None, "Adverse"],   # 4 non-null
    "e": ["Adverse", "Favorable", "Favorable", "Adverse", None, None],  # 4 non-null
})
ra = R._assoc(tt, "g", lambda x: x == "Adverse", "e", lambda x: x == "Adverse")
check("_assoc support = both-columns-non-null", ra["n"] == 3, str(ra["n"]))  # rows 0,1,2
check("_assoc counts the 2x2 correctly", ra["table"] == [[1, 1], [0, 1]], str(ra["table"]))

print("\n== _concordance_perm ==")
e = pd.Series(["Favorable", "Adverse", "Intermediate"] * 10)
cperf = R._concordance_perm(e, e.copy(), n_perm=2000, seed=0)
check("perfect concordance: agreement 1.0", cperf["agreement"] == 1.0)
check("perfect concordance: small perm p", cperf["p_perm"] < 0.01, str(cperf["p_perm"]))
shuf = pd.Series(list(e.sample(frac=1, random_state=1)))
crand = R._concordance_perm(e, shuf, n_perm=2000, seed=0)
check("shuffled concordance: p not tiny", crand["p_perm"] > 0.01, str(crand["p_perm"]))

print("\n== full validation on the real cohort ==")
ctx = amlmm.build_context(amlmm.Config(run_id="regrD"))
rep = R.run_validation(ctx)
cov, ts = rep["coverage"], rep["tests"]
print("   coverage:", cov)
check("anchored driver coverage in expected range (~150)", 120 <= cov["n_anchored_driver"] <= 200, str(cov["n_anchored_driver"]))
check("ELN coverage present (~70)", 50 <= cov["n_eln"] <= 90, str(cov["n_eln"]))
check("all 5 test sections present",
      set(ts) == {"anchored_eln_concordance", "adverse_driver_vs_eln_adverse",
                  "favorable_driver_vs_responder", "pLSC_vs_eln_adverse", "survival_by_expected_risk"})
check("ELN concordance computed on overlapping support", ts["anchored_eln_concordance"]["n"] > 0,
      str(ts["anchored_eln_concordance"]["n"]))
# adverse-driver test shares the (anchored & ELN) support with the concordance test -> same n,
# and that n cannot exceed the ELN coverage (the env-dependent-leak bug made it exceed it)
check("adverse-driver test support == concordance support (clean intersection)",
      ts["adverse_driver_vs_eln_adverse"]["n"] == ts["anchored_eln_concordance"]["n"],
      f"{ts['adverse_driver_vs_eln_adverse']['n']} vs {ts['anchored_eln_concordance']['n']}")
for k in ("adverse_driver_vs_eln_adverse", "favorable_driver_vs_responder", "pLSC_vs_eln_adverse"):
    check(f"{k} n does not exceed its label coverage",
          ts[k]["n"] <= max(cov["n_eln"], cov["n_responder"]), f"{k} n={ts[k]['n']}")
check("report carries the honesty caveat", "NOT trained predictors" in rep["caveat"])
md = open(ctx.path("VALIDATION.md"), encoding="utf-8").read()
check("VALIDATION.md renders coverage + tests", "Coverage" in md and "## Tests" in md)
# print the headline associations for the record
ec = ts["anchored_eln_concordance"]
print(f"   ELN concordance: agreement={ec['agreement']} n={ec['n']} perm_p={ec['p_perm']} underpowered={ec['underpowered']}")
print(f"   adverse-driver->ELN-adverse: OR={ts['adverse_driver_vs_eln_adverse']['odds_ratio']} "
      f"n={ts['adverse_driver_vs_eln_adverse']['n']} p={ts['adverse_driver_vs_eln_adverse']['p']}")
print(f"   p-LSC->ELN-adverse: OR={ts['pLSC_vs_eln_adverse']['odds_ratio']} "
      f"n={ts['pLSC_vs_eln_adverse']['n']} p={ts['pLSC_vs_eln_adverse']['p']}")

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
