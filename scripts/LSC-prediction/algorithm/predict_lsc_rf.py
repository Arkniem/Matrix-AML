#!/usr/bin/env python3
"""Reusable deployment of the saved LSC RandomForest model.

Usage:
    python3 predict_lsc_rf.py <sample_cellstate_counts.tsv> [--model LSC_RF_classifier.joblib] [--out predictions.tsv]

Input: TSV with a 'SampleID' column + one column per cell state holding per-sample
cell COUNTS (any cell-state vocabulary; the model aligns to its training states).
Frequencies are computed as count / total cells per sample before prediction.
"""
import argparse, os
import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("counts_tsv")
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "LSC_RF_classifier.joblib"))
ap.add_argument("--out", default=None)
a = ap.parse_args()

pay = joblib.load(a.model)
rf, feats, classes = pay["model"], pay["feature_states"], list(pay["model"].classes_)

cnt = pd.read_csv(a.counts_tsv, sep="\t")
if "SampleID" not in cnt.columns:
    raise SystemExit("Input must have a 'SampleID' column.")
dep = [c for c in cnt.columns if c != "SampleID"]
total = cnt[dep].astype(float).sum(axis=1)
freq = pd.DataFrame(index=cnt.index)
for s in feats:
    c = cnt[s].astype(float) if s in dep else 0.0
    freq[s] = np.where(total.values > 0, np.asarray(c) / np.where(total.values == 0, 1, total.values), 0.0)

proba = rf.predict_proba(freq[feats].values)
pred = np.array(classes)[np.argmax(proba, axis=1)]
out = pd.DataFrame({"SampleID": cnt["SampleID"], "PredictedClass": pred,
                    "MaxProb": proba.max(axis=1), "TotalCells": total.values})
for j, c in enumerate(classes):
    out[f"Prob_{c}"] = proba[:, j]
out["LowConfidence"] = out["MaxProb"] < 0.5
out["FewCells"] = out["TotalCells"] < 50

outp = a.out or os.path.splitext(a.counts_tsv)[0] + "_LSC_RF_predictions.tsv"
out.to_csv(outp, sep="\t", index=False)
print("Wrote:", outp)
print(out["PredictedClass"].value_counts().to_string())
