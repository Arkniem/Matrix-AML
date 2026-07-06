#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from amlmm.predictor import MutationPredictor

P = MutationPredictor.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutation_predictor.pkl"))
cf = None
for (m, mod), mm in P.models.items():
    if mod == "Composition":
        cf = mm.features; break
print("Composition feature names (%d):" % len(cf), cf[:8])

A = "/data/salomonis2/LabFiles/Frank-Li/scTriangulate/Hs_AML_UDON/output/cellHarmony_lite_assignments.txt"
counts = {}
with open(A, encoding="utf-8", errors="replace") as fh:
    h = fh.readline().rstrip("\n").split("\t"); bi, si = h.index("CellBarcode"), h.index("Hs-BM-titrated-reference-centroid")
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) <= max(bi, si):
            continue
        s = p[bi].split(".", 1)[1] if "." in p[bi] else None
        if s:
            counts.setdefault(s, {}).setdefault(p[si], 0); counts[s][p[si]] += 1

two = sorted(counts)[:2]
for s in two:
    tot = sum(counts[s].values()) or 1
    comp = pd.Series({k: v / tot for k, v in counts[s].items()})
    aligned = [f for f in cf if f in comp.index]
    print("\n%s: %d cells, comp states %d, aligned-to-features %d/%d, top states: %s"
          % (s, tot, len(comp), len(aligned), len(cf),
             ", ".join("%s=%.2f" % (k, comp[k]) for k in comp.sort_values(ascending=False).index[:4])))
    for m in ["NPM1", "FLT3", "complex"]:
        mm = P.models.get((m, "Composition"))
        if mm:
            print("   %-8s Composition.score = %.4f  (weight in predictor=%.3f)"
                  % (m, mm.score(comp), P.weights.get(m, {}).get("Composition", 0)))
print("DIAG OK")
