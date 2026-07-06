#!/usr/bin/env python3
"""Train the bake-off WINNER (RandomForest on all state frequencies) on the 27
labeled samples and persist a self-contained, reusable model payload. Verify it
reproduces the deployed 397-sample calls exactly."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = "/Users/saljh8/Dropbox/Collaborations/Grimes/UDON"
OUT = f"{BASE}/deployed_results_combined_annotations_sample_level"
TRAIN = f"{BASE}/iterative_logistic_results/sample_cell_frequencies.tsv"

tr = pd.read_csv(TRAIN, sep="\t")
feature_states = [c for c in tr.columns if c not in ("SampleID", "Class", "TotalCells")]
Xtr = tr[feature_states].astype(float).values
ytr = tr["Class"].astype(str).values

rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=1)
rf.fit(Xtr, ytr)

payload = {
    "model": rf,
    "model_type": "RandomForestClassifier",
    "feature_states": feature_states,                 # ordered 82 cell states (model inputs)
    "feature_space": "per-sample cell-state frequency (state_count / total_cells_in_sample)",
    "classes": list(rf.classes_),
    "class_order": ["p-LSC", "m-LSC", "p+m-LSC"],
    "training_n": int(len(tr)),
    "training_class_counts": tr["Class"].value_counts().to_dict(),
    "rf_params": {"n_estimators": 400, "class_weight": "balanced_subsample", "random_state": 1},
    "selection_basis": "honest paired nested CV (30x3-fold) bake-off across 6 model families",
    "honest_cv_performance": {"WeightedObjective": 0.696, "BalancedAccuracy": 0.589,
                              "MacroF1": 0.516, "WeightedObjective_std": 0.033},
    "weighted_objective_def": "0.45*p-LSC_F1 + 0.45*m-LSC_F1 + 0.10*p+m-LSC_F1",
    "caveats": [
        "No normal/control class: only separates the 3 AML LSC subtypes.",
        "Weak absolute performance (~0.59 balanced acc, 3-class) -> soft calls.",
        "Under-calls p+m-LSC; use aggregate-logistic model if dual subtype matters.",
        "Predict only on samples with sufficient cells (>=50).",
    ],
}
model_path = f"{OUT}/LSC_RF_classifier.joblib"
joblib.dump(payload, model_path)
print("[INFO] saved:", model_path)

# ---- verify reproduction of deployed calls ----
cnt = pd.read_csv(f"{OUT}/sample_cellstate_counts.tsv", sep="\t")
dep_states = [c for c in cnt.columns if c != "SampleID"]
total = cnt[dep_states].astype(float).sum(axis=1)
freq = pd.DataFrame(index=cnt.index)
for s in feature_states:
    c = cnt[s].astype(float) if s in dep_states else 0.0
    freq[s] = np.where(total.values > 0, np.asarray(c) / np.where(total.values == 0, 1, total.values), 0.0)
pred = np.array(rf.classes_)[np.argmax(rf.predict_proba(freq[feature_states].values), axis=1)]
ref = pd.read_csv(f"{OUT}/prediction_subtype_classifications_RF.tsv", sep="\t").set_index("SampleID")
new = pd.Series(pred, index=cnt["SampleID"])
agree = (new.reindex(ref.index).values == ref["PredictedClass"].values).mean()
print(f"[INFO] saved model reproduces deployed calls: {agree:.1%} ({len(ref)} samples)")
assert agree == 1.0, "Saved model does NOT reproduce deployed calls!"
print("[INFO] QC PASS")
