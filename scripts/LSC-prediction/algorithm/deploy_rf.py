#!/usr/bin/env python3
"""
Deploy the honest bake-off WINNER (RandomForest on all state frequencies) to the
397 Dataset:Sample profiles. Train on the 27 labeled samples; align deployment
frequencies to the training state vocabulary; output probabilistic LSC-subtype calls.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = "/Users/saljh8/Dropbox/Collaborations/Grimes/UDON"
OUT = f"{BASE}/deployed_results_combined_annotations_sample_level"
TRAIN = f"{BASE}/iterative_logistic_results/sample_cell_frequencies.tsv"
COUNTS = f"{OUT}/sample_cellstate_counts.tsv"

# ---- training data (frequencies already computed) ----
tr = pd.read_csv(TRAIN, sep="\t")
state_cols = [c for c in tr.columns if c not in ("SampleID", "Class", "TotalCells")]
Xtr = tr[state_cols].astype(float).values
ytr = tr["Class"].astype(str).values
print(f"[INFO] train: {Xtr.shape[0]} samples x {Xtr.shape[1]} states")

# ---- deployment counts -> frequencies over the SAME state vocabulary ----
cnt = pd.read_csv(COUNTS, sep="\t")
dep_states = [c for c in cnt.columns if c != "SampleID"]
total = cnt[dep_states].astype(float).sum(axis=1)           # total cells per sample
missing = [s for s in state_cols if s not in dep_states]
extra = [s for s in dep_states if s not in state_cols]
print(f"[INFO] training states missing from deployment (filled 0): {missing}")
print(f"[INFO] deployment-only states ignored by model ({len(extra)}): {extra}")

freq = pd.DataFrame(index=cnt.index)
for s in state_cols:
    c = cnt[s].astype(float) if s in dep_states else 0.0
    freq[s] = np.where(total.values > 0, np.asarray(c) / np.where(total.values == 0, 1, total.values), 0.0)
Xde = freq[state_cols].astype(float).values

# ---- fit RF on all training data, deploy ----
rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=1)
rf.fit(Xtr, ytr)
classes = list(rf.classes_)
proba = rf.predict_proba(Xde)
pred = np.array(classes)[np.argmax(proba, axis=1)]
maxp = proba.max(axis=1)

out = pd.DataFrame({"SampleID": cnt["SampleID"], "PredictedClass": pred, "MaxProb": maxp})
for j, c in enumerate(classes):
    out[f"Prob_{c}"] = proba[:, j]
out["TotalCells"] = total.values

# annotate with Annotation/Dataset/Sample
ann = pd.read_csv(f"{BASE}/cellHarmony-datasets/final/combined_annotations.txt", sep="\t",
                  dtype=str, usecols=["Sample", "Annotation", "Dataset"], keep_default_na=False)
ann["SampleID"] = ann["Dataset"] + ":" + ann["Sample"]
meta = ann.drop_duplicates("SampleID").set_index("SampleID")[["Sample", "Dataset", "Annotation"]]
out = out.merge(meta, left_on="SampleID", right_index=True, how="left")
out["LowConfidence"] = out["MaxProb"] < 0.5
out["FewCells"] = out["TotalCells"] < 50

cols = ["SampleID", "Sample", "Dataset", "Annotation", "TotalCells", "PredictedClass",
        "MaxProb", "LowConfidence", "FewCells"] + [f"Prob_{c}" for c in classes]
out = out[cols].sort_values(["PredictedClass", "Annotation"])
op = f"{OUT}/prediction_subtype_classifications_RF.tsv"
out.to_csv(op, sep="\t", index=False)

print("\n=== RF predicted class counts (397 samples) ===")
print(out["PredictedClass"].value_counts().to_string())
print("\n=== RF vs Control/AML ===")
g = np.where(out["Annotation"].str.contains("Control", case=False, na=False), "Control", "AML/other")
print(pd.crosstab(g, out["PredictedClass"]).to_string())
print(f"\n[INFO] low-confidence (MaxProb<0.5): {int(out.LowConfidence.sum())}; few-cells(<50): {int(out.FewCells.sum())}")

# ---- agreement vs the v3 aggregate deployment ----
v3 = pd.read_csv(f"{OUT}/prediction_subtype_classifications.tsv", sep="\t",
                 keep_default_na=False, na_values=[""])[["SampleID", "PredictedClass"]]
v3 = v3.rename(columns={"PredictedClass": "v3_AGG"})
cmp = out[["SampleID", "PredictedClass"]].rename(columns={"PredictedClass": "RF"}).merge(v3, on="SampleID")
agree = (cmp["RF"] == cmp["v3_AGG"]).mean()
print(f"\n[INFO] RF vs v3-AGG agreement: {agree:.1%} of 397 samples")
print(pd.crosstab(cmp["RF"], cmp["v3_AGG"], dropna=False).to_string())
cmp.to_csv(f"{OUT}/RF_vs_AGG_call_comparison.tsv", sep="\t", index=False)
print(f"\nWrote: {op}")
print(f"Wrote: {OUT}/RF_vs_AGG_call_comparison.tsv")
