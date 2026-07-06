# Patient triage — CCHMC::0018_Af_N1c
(atlas annotation: NPM1c · dataset: CCHMC · KB kb-2026.06)

## Witness reads
- **composition** (independent / honest_cv, conf 0.5, weight 0.85): predicts NPM1 (prob 0.9161)
- **genetic** (independent / deterministic_fact, conf 0.6, weight 0.85): present: ['complex', 'NPM1']; targetable: ['NPM1']
- **cell-state/UDON** (rna_derived / discovery, conf 0.6, weight 0.7): programs: ['P3', 'P14', 'P7', 'P6', 'P2']

## Decision (high confidence)
- **Subtype call:** NPM1 (genetic-anchored: direct mutation observation)  (concordance 0.885)
- **Per-witness consistency:** {'composition': 'agree', 'genetic': 'agree', 'cell-state/UDON': 'conflict'}
- **Conflicts:** cell-state/UDON→Inv16

### Ranked therapy hypotheses (knowledge-grounded)
- NPM1 → menin inhibitor (revumenib)  [guideline; NCCN-2024]
- complex → HMA + venetoclax; consider allogeneic transplant / trial  [heuristic; lit]

### Recommended validations
- mutation: targeted DNA/RNA sequencing (NGS panel)
- subtype: flow cytometry immunophenotyping
- complex: conventional karyotype / SNP-array cytogenetics

### Rationale
Leading hypothesis: NPM1 (genetic-anchored: direct mutation observation). Concordance 0.89. Conflicts: cell-state/UDON→Inv16. 