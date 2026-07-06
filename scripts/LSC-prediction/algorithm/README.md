# LSC subtype prediction — combined_annotations.txt (sample level)

Predict AML leukemic-stem-cell (LSC) subtype — **p-LSC** (primitive), **m-LSC**
(monocytic), **p+m-LSC** (dual) — for each sample in
`cellHarmony-datasets/final/combined_annotations.txt`, from per-sample
cell-state composition.

## Input
- `combined_annotations.txt`: 3,208,224 cells × {CellBarcode, cell-state call
  (`Hs-BM-titrated-reference-centroid`), Sample, Annotation, Dataset}.
- Sample unit = **`Dataset:Sample`** (composite key; disambiguates 8 sample-name
  collisions across datasets, matching the prior `deployed_results/` convention).
- → 397 samples × 90 cell states. Frequencies = state count / total cells per sample.

## Which model? (honest bake-off)
The three scripts and an older forward-selection model were compared on the 27
labeled training samples (12 p-LSC / 8 m-LSC / 7 p+m-LSC) under **paired, repeated
nested cross-validation (30×3-fold), with all feature selection done INSIDE each
fold** so the CV is unbiased. Metric = project WeightedObjective
(0.45·p-LSC F1 + 0.45·m-LSC F1 + 0.10·p+m-LSC F1).

| Model | WeightedObjective | BalancedAcc | MacroF1 | std |
|---|---|---|---|---|
| **RandomForest (all states)** — *winner, deployed* | **0.696** | **0.589** | 0.516 | **0.033** |
| Aggregate-lineage logistic (LSC_classifier2/3, k=17/20) | 0.649 | 0.551 | **0.558** | 0.065 |
| Logistic (all states) | 0.636 | 0.527 | 0.517 | 0.077 |
| Aggregate-lineage logistic (k=6/6) | 0.611 | 0.497 | 0.502 | 0.072 |
| Forward-selection logistic (≤6 states) | 0.586 | 0.506 | 0.497 | 0.077 |
| Forward-selection logistic (≤4 states) | 0.582 | 0.502 | 0.496 | 0.077 |

- **The forward-selection model's previously-reported 0.85 balanced accuracy was
  selection bias.** Evaluated honestly it is the *worst* family.
- **RandomForest is best** by WeightedObjective + BalancedAccuracy + lowest
  variance, and is independently corroborated (v1's own run: RF BalAcc 0.60).
- `LSC_classifier2.py` ≡ `LSC_classifier3.py` (same model); only v3 has a
  prediction mode. The aggregate logistic is a close second and is **better
  balanced across classes (highest MacroF1)** — notably better at calling p+m-LSC.

## Model files
- `LSC_RF_classifier.joblib` — **final model** (RandomForest winner). Self-contained
  payload: fitted model, ordered 82 training cell states, classes, RF params, honest
  CV performance, caveats.
- `predict_lsc_rf.py` — standalone re-deployment:
  `python3 predict_lsc_rf.py <sample_cellstate_counts.tsv>` → per-sample LSC calls +
  probabilities (aligns any cell-state vocabulary to the model's training states).
- `LSC_aggregate_logistic_classifier.joblib` — **alternative** model (LSC_classifier2/3
  aggregate logistic, k=17/20). Lower WeightedObjective but better balanced across
  classes (highest MacroF1); prefer it if calling p+m-LSC matters. Deploy with
  `LSC_classifier3.py <counts.tsv> --load-model LSC_aggregate_logistic_classifier.joblib`.

## Deliverables
- `prediction_subtype_classifications_RF.tsv` — **RECOMMENDED** call set (RF winner):
  PredictedClass, per-class probabilities, MaxProb, LowConfidence/FewCells flags,
  Annotation/Dataset.
- `prediction_subtype_classifications.tsv` — v3 aggregate-logistic calls (comparison;
  raw classifier output, incl. NULL = below-threshold).
- `prediction_aggregate_features.tsv` — v3 aggregate features per sample.
- `prediction_summary_annotated.tsv` — v3 calls + annotation/cell-count/probabilities.
- `RF_vs_AGG_call_comparison.tsv` — per-sample RF vs aggregate call (agreement 69.8%).
- `sample_cellstate_counts.tsv` — 397×90 count matrix (model input; QC: all
  3,208,224 cells accounted for).
- `honest_model_bakeoff.py` / `honest_model_bakeoff_results.tsv` — bake-off code + table.
- `deploy_rf.py` — RF training + deployment script.

## RF calls (397 samples)
p-LSC 288 · m-LSC 93 · p+m-LSC 16.

## Caveats (read before use)
1. **No normal/control class.** The model only separates the 3 AML LSC subtypes, so
   the 45 Control samples are force-labeled (34 p-LSC, 11 m-LSC) — not meaningful.
2. **Weak absolute performance** (~0.59 balanced accuracy, 3-class). Treat calls as
   soft/probabilistic, not ground truth. **148/397 calls are low-confidence
   (MaxProb<0.5).**
3. **RF under-calls p+m-LSC** (16 vs the aggregate model's 83). If detecting the
   dual p+m-LSC subtype matters biologically, prefer the aggregate-logistic calls
   (better MacroF1) or require concordance between the two.
4. **7 samples have <50 cells** (down to 1 cell) — degenerate; exclude.
5. Tiny training set (n=27); model-ranking margins are within ~1 SD.
