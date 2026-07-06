# MATRIX-AML — Detailed Prediction Report
**Sample:** `P13_Diagnosis.Heidelberg`  |  **Cohort:** Trumpp/Waclawiczek (external, Cell Stem Cell 2025)  |  **Generated:** 2026-06-30 21:00
**Ground truth (Table S4 known drivers):** DNMT3A, FLT3, NPM1

### Specimen gate (healthy-vs-diseased)
- Call: **diseased**  (p_diseased = 0.9731)

### Modalities used for this prediction
- **Composition** — measured (cellHarmony cell-state fractions)
- **RNA** — measured (pseudobulk gene expression)
- **ADT** — imputed from RNA (rna2adt bundle, calibrated)
- **GRN** — imputed from RNA (rna2grn bundle)
- **Lipid** — imputed from RNA (rna2lipid bundle)
- **LSC** — computed (RandomForest on cell-state frequencies)
- **Cell-comm** — computed (fastComm on the cells; CellChatDB ligand-receptor pairs)
- _Deferred (not available for this external sample):_ Metabolite (NOT AVAILABLE (bundle off-cluster))

### Mutation panel (all drivers, sorted by probability)
| driver | prob | call | truth | result | modality coverage | confidence |
|---|---|---|---|---|---|---|
| DNMT3A ⟵ known | 0.886 | present | present | ✅ TP | 100% | ok |
| GATA2 | 0.854 | present | absent | ⚠️ FP | 100% | abstain: underpowered (n+<3) |
| NPM1 ⟵ known | 0.836 | present | present | ✅ TP | 100% | ok |
| trisomy8 | 0.798 | present | absent | ⚠️ FP | 99% | abstain: underpowered (n+<3) |
| NRAS | 0.778 | present | absent | ⚠️ FP | 86% | abstain: underpowered (n+<3) |
| TET2 | 0.771 | present | absent | ⚠️ FP | 91% | ok |
| complex | 0.748 | present | absent | ⚠️ FP | 45% | abstain: low modality coverage (45% of weight present) |
| PTPN11 | 0.716 | present | absent | ⚠️ FP | 100% | abstain: underpowered (n+<3) |
| CEBPA | 0.677 | present | absent | ⚠️ FP | 74% | abstain: underpowered (n+<3) |
| ASXL1 | 0.629 | present | absent | ⚠️ FP | 100% | ok |
| IDH1 | 0.562 | present | absent | ⚠️ FP | 100% | abstain: underpowered (n+<3) |
| FLT3-ITD | 0.558 | present | absent | ⚠️ FP | 100% | abstain: underpowered (n+<3) |
| KIT | 0.538 | present | absent | ⚠️ FP | 100% | abstain: underpowered (n+<3) |
| IDH2 | 0.525 | present | absent | ⚠️ FP | 100% | abstain: underpowered (n+<3) |
| inv16 | 0.504 | present | absent | ⚠️ FP | 89% | abstain: underpowered (n+<3) |
| inv(16)_CBFB-MYH11 | 0.504 | present | absent | ⚠️ FP | 89% | abstain: underpowered (n+<3) |
| kmt2a | 0.404 | absent | absent | ✅ TN | 80% | abstain: underpowered (n+<3) |
| KMT2A-rearrangement | 0.404 | absent | absent | ✅ TN | 80% | abstain: underpowered (n+<3) |
| FLT3-TKD | 0.4 | absent | absent | ✅ TN | 100% | abstain: underpowered (n+<3) |
| WT1 | 0.332 | absent | absent | ✅ TN | 100% | abstain: underpowered (n+<3) |
| FLT3 ⟵ known | 0.315 | absent | present | ❌ FN | 86% | ok |
| KRAS | 0.302 | absent | absent | ✅ TN | 100% | abstain: underpowered (n+<3) |
| TP53 | 0.256 | absent | absent | ✅ TN | 73% | ok |
| del5 | 0.233 | absent | absent | ✅ TN | 43% | abstain: underpowered (n+<3) |
| del7 | 0.165 | absent | absent | ✅ TN | 65% | abstain: underpowered (n+<3) |
| RUNX1 | 0.119 | absent | absent | ✅ TN | 3% | abstain: underpowered (n+<3) |

### Evidence breakdown for the known drivers
**DNMT3A** — probability 0.886 (present), coverage 100%, ok

| modality | weight | oriented score (0-1) | cohort OOF AUC |
|---|---|---|---|
| GRN | 0.496 | 0.837 | 0.752 |
| Cell-comm | 0.44 | 0.936 | 0.735 |
| RNA | 0.065 | 0.925 | 0.706 |
| Composition | 0.0 | 0.87 | 0.527 |
| ADT | 0.0 | 0.468 | 0.594 |
| Lipid | 0.0 | 0.884 | 0.651 |
| LSC | 0.0 | 1.0 | 0.525 |

**NPM1** — probability 0.836 (present), coverage 100%, ok

| modality | weight | oriented score (0-1) | cohort OOF AUC |
|---|---|---|---|
| RNA | 0.561 | 0.787 | 0.891 |
| GRN | 0.318 | 0.886 | 0.864 |
| Cell-comm | 0.121 | 0.933 | 0.8 |
| Composition | 0.0 | 0.751 | 0.697 |
| ADT | 0.0 | 0.687 | 0.793 |
| Lipid | 0.0 | 0.848 | 0.82 |
| LSC | 0.0 | 1.0 | 0.551 |

**FLT3** — probability 0.315 (absent), coverage 86%, ok

| modality | weight | oriented score (0-1) | cohort OOF AUC |
|---|---|---|---|
| ADT | 0.826 | 0.296 | 0.836 |
| Lipid | 0.031 | 0.823 | 0.755 |
| RNA | 0.0 | 0.856 | 0.707 |
| Composition | 0.0 | 0.501 | 0.601 |
| GRN | 0.0 | 0.886 | 0.74 |
| LSC | 0.0 | 1.0 | 0.637 |
| Cell-comm | 0.0 | 0.913 | 0.635 |

### How to read this
- **Oriented score** = the sample's percentile within the cohort for that modality's model (0.5 = cohort median; >0.5 leans mutation-present). **Probability** = weight-blend of the available modalities' scores, weights renormalised over what's present.
- **Confidence = abstain** means the deployed model lacks enough evidence (too few cohort positives, or its key modalities — Metabolite/Cell-comm — are not available for this external sample) and should NOT be read as a confident call.
- Imputed modalities (ADT/GRN/Lipid) are derived from this sample's RNA, not independently measured; they add cohort-tuned structure but cannot exceed the RNA's information.