# Patient triage — WashU::10DD-1002__Diagnosis
(atlas annotation: TP53 U2AF1 · dataset: WashU · KB kb-2026.06)

## Witness reads
- **composition** (independent / honest_cv, conf 0.5, weight 0.85): predicts FLT3 (prob 0.7819)
- **genetic** (independent / deterministic_fact, conf 0.6, weight 0.85): present: ['del7', 'TP53']; targetable: []
- **cell-state/UDON** (rna_derived / discovery, conf 0.6, weight 0.7): programs: ['P8', 'P2', 'P9', 'P6', 'P1', 'P10', 'P7', 'P3']

## Decision (medium confidence)
- **Subtype call:** TP53 (genetic-anchored: direct mutation observation)  (concordance 0.581)
- **Per-witness consistency:** {'composition': 'conflict', 'genetic': 'agree', 'cell-state/UDON': 'agree'}
- **Conflicts:** composition→FLT3

### Ranked therapy hypotheses (knowledge-grounded)
- TP53 → HMA + venetoclax; consider trial (eprenetapopt) and early allogeneic transplant  [heuristic; lit]
- del7 → HMA + venetoclax; consider allogeneic transplant / trial  [heuristic; lit]

### Recommended validations
- mutation: targeted DNA/RNA sequencing (NGS panel)
- subtype: flow cytometry immunophenotyping
- del7: FISH for -7/del(7q) and conventional karyotype

### Rationale
Leading hypothesis: TP53 (genetic-anchored: direct mutation observation). Concordance 0.58. Conflicts: composition→FLT3. 