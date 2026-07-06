# Patient triage — WashU::10DD-1002__Diagnosis
(atlas annotation: TP53 U2AF1 · dataset: WashU · KB kb-2026.07)

## Witness reads
- **composition** (independent / honest_cv, conf 0.5, weight 0.425): predicts FLT3 (prob 0.7819)
- **genetic** (independent / deterministic_fact, conf 0.6, weight 0.85): present: ['del7', 'TP53']; targetable: []
- **cell-state/UDON** (rna_derived / discovery, conf 0.6, weight 0.7): programs: ['P8', 'P2', 'P9', 'P1', 'P10', 'P6', 'P3', 'P7']
- **LSC** (rna_derived / classifier_call, conf 0.5, weight 0.5): LSC architecture call: p-LSC (MaxProb 0.718, low_confidence=False, few_cells=False); upstream RF honest balanced-acc ~0.59 — corroborating stemness/risk context, not ground truth.
- **surfaceome/ADT** (imputed_from_RNA / descriptive_aggregate, conf 0.5, weight 0.4): Elevated (imputed) surface markers vs cohort (z>=1): CD235a(z1.806), CD279(z1.626), CD85g(z1.572), CD25(z1.492), Cadherin(z1.275), CD200(z1.264). Imputed-from-RNA hypotheses; confirm by flow.
- **metabolic** (imputed_from_RNA / descriptive_aggregate, conf 0.45, weight 0.35): metabolic: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.463): Phosphorylcholine+, Adenosine triphosphate (ATP)+, Fructose 1,6-Bisphosphate-, Lenticin-, 
- **lipid** (imputed_from_RNA / descriptive_aggregate, conf 0.45, weight 0.35): lipid: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.557): PE O-42:1|PE O-24:0_18:1+, Hex2Cer 44:2;2O|Hex2Cer 18:1;2O/26:1-, PC 37:3|PC 19:1_18:2+, HexCe
- **GRN-regulon** (imputed_from_RNA / descriptive_aggregate, conf 0.4, weight 0.3): Top active regulators (imputed GRN — fidelity UNKNOWN, no held-out metric): SPI1, YBX1, FOS, CEBPD, RUNX1. Discovery context only.
- **cell-communication** (independent / descriptive_aggregate, conf 0.5, weight 0.5): Dominant signaling axes (independent fastComm L-R inference): PPIA->BSG, APP->CD74, MIF->CD44+CD74, HLA-B->CD8B, HLA-A->CD8B; 9099 called interactions.

## Decision (high confidence)
- **Subtype call:** TP53 (genetic-anchored: direct mutation observation)  (concordance 0.902)
- **Per-witness consistency:** {'composition': 'conflict', 'genetic': 'agree', 'cell-state/UDON': 'agree'}
- **Conflicts:** composition→FLT3

### Ranked therapy hypotheses (knowledge-grounded)
- TP53 → HMA + venetoclax; consider trial (eprenetapopt) and early allogeneic transplant  [heuristic; lit]
- del7 → HMA + venetoclax; consider allogeneic transplant / trial  [heuristic; lit]

### Recommended validations
- mutation: targeted DNA/RNA sequencing (NGS panel)
- subtype: flow cytometry immunophenotyping
- del7: FISH for -7/del(7q) and conventional karyotype
- surface_marker: flow cytometry for the named surface marker
- lsc: ex-vivo drug-sensitivity assay
- metabolite: targeted metabolomics / LC-MS confirmation of the imputed metabolite shift
- lipid_profile: untargeted lipidomics / mass-spectrometry confirmation of the imputed lipid-class shift

### Descriptive / discovery context (corroborating; non-voting)
- _GRN-regulon_ (grn): Top active regulators (imputed GRN — fidelity UNKNOWN, no held-out metric): SPI1, YBX1, FOS, CEBPD, RUNX1. Discovery context only.
- _LSC_ (lsc): LSC architecture call: p-LSC (MaxProb 0.718, low_confidence=False, few_cells=False); upstream RF honest balanced-acc ~0.59 — corroborating stemness/risk context, not ground truth.
- _cell-communication_ (cell_comm): Dominant signaling axes (independent fastComm L-R inference): PPIA->BSG, APP->CD74, MIF->CD44+CD74, HLA-B->CD8B, HLA-A->CD8B; 9099 called interactions.
- _lipid_ (lipid): lipid: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.557): PE O-42:1|PE O-24:0_18:1+, Hex2Cer 44:2;2O|Hex2Cer 18:1;2O/26:1-, PC 37:3|PC 19:1_18:2+, HexCer 36:1;2O|HexCer 18:1;2O/18:0-, FA 28:6;
- _metabolic_ (metabolic): metabolic: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.463): Phosphorylcholine+, Adenosine triphosphate (ATP)+, Fructose 1,6-Bisphosphate-, Lenticin-, Guanosine Diphosphate (GDP)+, Citramalic
- _surfaceome/ADT_ (surfaceome): Elevated (imputed) surface markers vs cohort (z>=1): CD235a(z1.806), CD279(z1.626), CD85g(z1.572), CD25(z1.492), Cadherin(z1.275), CD200(z1.264). Imputed-from-RNA hypotheses; confirm by flow.

### Deliberation (Phase C, continuous)
- **Rounds:** 2 (converged)
- **Baseline → final leading:** TP53 → TP53 (changed: False; final genetically confirmed: True)
- **Concordance baseline → final:** 0.822 → 0.902
- **Groupthink warning:** False

### Rationale
Leading hypothesis: TP53 (genetic-anchored: direct mutation observation). Concordance 0.90. Conflicts: composition→FLT3. 