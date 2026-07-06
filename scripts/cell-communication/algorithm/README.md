# Per-sample fastComm cell-cell communication (pseudobulks as cells)

Generated 2026-06-12 17:06:04 by `run_fastcomm_per_sample.py` (2.1 min). One h5ad per
sample encoding receptor-ligand cell-cell signaling, computed with the same
fastComm scoring cellHarmony-web runs per sample.

## Inputs / method
- `pseudobulk_counts_hashed.h5ad` (12255 pseudobulks x 35702 genes); each
  pseudobulk = one cell-state of one specimen, treated as a cell.
- RNA normalized **CP10k + log1p** (as cellHarmony feeds fastComm).
- Per sample: `run_fastcomm(state_key='Hs-BM-titrated-reference-centroid', lr_sources=CellChatDB,
  response_matrix=None, species=human, min_cells=1, min_lr_expression_score=0.2,
  max_lr_candidates_per_state_pair=5, include_self_edges=False)`. min_cells=1 is the
  only change vs cellHarmony-web (pseudobulks = 1 cell/state).

## Per-sample h5ad (`per_sample/fastcomm_<Sample>.h5ad`)
- `obs` = interactions (index `sender|ligand|receptor|receiver`); cols sender_state,
  receiver_state, ligand, receptor, lr_key, pathway, interaction_class, state_pair,
  lr_pair, fastcomm_score.
- `var` = numeric fastComm metrics; `X` = interactions x metrics (fastcomm_score,
  lr_expression_score, receiver_response_score, percentiles, ...). Primary = fastcomm_score.
- `uns` = sample/Dataset/Annotation, cell_states (+sizes), **state_pair_strength**
  (sender x receiver aggregated fastcomm_score = cell-cell signaling matrix),
  state_pair_n_edges, fastcomm_params, fastcomm_summary.

## Cross-sample unsupervised analysis
`combined_sample_by_interaction.h5ad` = samples x interaction_id matrix of
fastcomm_score (sparse; 0 = not detected), obs carries Dataset/Annotation. Ready
for clustering AML vs control by communication profile. `sample_manifest.tsv`
lists every sample's status / n_states / n_edges.
