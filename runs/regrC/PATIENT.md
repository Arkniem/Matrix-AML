# Patient triage — CCHMC::0018_Af_N1c
(atlas annotation: NPM1c · dataset: CCHMC · KB kb-2026.07)

## Witness reads
- **composition** (independent / honest_cv, conf 0.5, weight 0.85): predicts NPM1 (prob 0.9161)
- **genetic** (independent / deterministic_fact, conf 0.6, weight 0.85): present: ['complex', 'NPM1']; targetable: ['NPM1']
- **cell-state/UDON** (rna_derived / discovery, conf 0.6, weight 0.35): programs: ['P3', 'P14', 'P6', 'P7', 'P2']
- **LSC** (rna_derived / classifier_call, conf 0.1, weight 0.0): LSC architecture call: p-LSC (MaxProb 0.39, low_confidence=True, few_cells=False); upstream RF honest balanced-acc ~0.59 — corroborating stemness/risk context, not ground truth.
- **surfaceome/ADT** (imputed_from_RNA / descriptive_aggregate, conf 0.5, weight 0.4): Elevated (imputed) surface markers vs cohort (z>=1): CD90(z1.219). Imputed-from-RNA hypotheses; confirm by flow.
- **metabolic** (imputed_from_RNA / descriptive_aggregate, conf 0.45, weight 0.35): metabolic: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.538): Leucine methyl ester-, 6-Methylnicotinamide+, epsilon-(gamma-Glutamyl)lysine+, Homostachyd
- **lipid** (imputed_from_RNA / descriptive_aggregate, conf 0.45, weight 0.35): lipid: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.651): DG 38:5|DG 18:2_20:3-, FA 28:1+, Cer 44:1;2O|Cer 18:1;2O/26:0+, PC 38:4|PC 18:0_20:4-, TG 58:8
- **GRN-regulon** (imputed_from_RNA / descriptive_aggregate, conf 0.4, weight 0.3): Top active regulators (imputed GRN — fidelity UNKNOWN, no held-out metric): SPI1, ELF1, RUNX1, ELF2, ATF6. Discovery context only.
- **cell-communication** (independent / descriptive_aggregate, conf 0.5, weight 0.5): Dominant signaling axes (independent fastComm L-R inference): CD55->ADGRE5, NAMPT->ITGA5+ITGB1, CD69->KLRB1, MIF->CD44+CD74, MIF->CD74+CXCR4; 2172 called interactions.

## Decision (high confidence)
- **Subtype call:** NPM1 (genetic-anchored: direct mutation observation)  (concordance 0.911)
- **Per-witness consistency:** {'composition': 'agree', 'genetic': 'agree', 'cell-state/UDON': 'conflict'}
- **Conflicts:** cell-state/UDON→Inv16

### Ranked therapy hypotheses (knowledge-grounded)
- NPM1 → menin inhibitor (revumenib)  [guideline; NCCN-2024]
- complex → HMA + venetoclax; consider allogeneic transplant / trial  [heuristic; lit]

### Recommended validations
- mutation: targeted DNA/RNA sequencing (NGS panel)
- subtype: flow cytometry immunophenotyping
- complex: conventional karyotype / SNP-array cytogenetics
- surface_marker: flow cytometry for the named surface marker
- metabolite: targeted metabolomics / LC-MS confirmation of the imputed metabolite shift
- lipid_profile: untargeted lipidomics / mass-spectrometry confirmation of the imputed lipid-class shift

### Descriptive / discovery context (corroborating; non-voting)
- _GRN-regulon_ (grn): Top active regulators (imputed GRN — fidelity UNKNOWN, no held-out metric): SPI1, ELF1, RUNX1, ELF2, ATF6. Discovery context only.
- _LSC_ (lsc): LSC architecture call: p-LSC (MaxProb 0.39, low_confidence=True, few_cells=False); upstream RF honest balanced-acc ~0.59 — corroborating stemness/risk context, not ground truth.
- _cell-communication_ (cell_comm): Dominant signaling axes (independent fastComm L-R inference): CD55->ADGRE5, NAMPT->ITGA5+ITGB1, CD69->KLRB1, MIF->CD44+CD74, MIF->CD74+CXCR4; 2172 called interactions.
- _lipid_ (lipid): lipid: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.651): DG 38:5|DG 18:2_20:3-, FA 28:1+, Cer 44:1;2O|Cer 18:1;2O/26:0+, PC 38:4|PC 18:0_20:4-, TG 58:8|TG 18:1_18:2_22:5+, PC 44:2|PC 18:1_26:
- _metabolic_ (metabolic): metabolic: most-distinctive features (imputed, min held-out Spearman 0.3; median fidelity of reported 0.538): Leucine methyl ester-, 6-Methylnicotinamide+, epsilon-(gamma-Glutamyl)lysine+, Homostachydrine-, Prolinamide-, Neopterin-.
- _surfaceome/ADT_ (surfaceome): Elevated (imputed) surface markers vs cohort (z>=1): CD90(z1.219). Imputed-from-RNA hypotheses; confirm by flow.

### Deliberation (Phase C, continuous)
- **Deliberation rounds:** 1 (converged)
- **Baseline → final leading:** NPM1 → NPM1 (changed: False; final genetically confirmed: True)
- **Concordance baseline → final:** 0.837 → 0.911
- **Groupthink warning:** False

### Rationale
Leading hypothesis: NPM1 (genetic-anchored: direct mutation observation). Concordance 0.91. Conflicts: cell-state/UDON→Inv16. 