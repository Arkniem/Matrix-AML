# amlmm — AML multimodal base pipeline

A deterministic, reproducible pipeline over the AML multimodal atlas, structured so
LLM agents can be slotted in at named decision points later (see
[docs/AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md)). Runs **as per-config LSF jobs
on the cluster** and in-process locally for development.

## What it does (one config = one job)
`feasibility → assemble_features → classify → cluster_explore → report`

1. **feasibility** — per-target trainability triage (labeled n, usable classes,
   cohort confounding via Cramér's V, verdict). Stops you modeling untrainable targets.
2. **assemble_features** — build `X/y/groups` for a target from chosen feature blocks
   (`composition` = cell frequency; modality blocks = n_cells-weighted pseudobulk means,
   imputed modalities filtered to held-out Spearman ≥ 0.3).
3. **classify** — honest **group-aware nested CV** (donor-grouped or leave-one-cohort-out;
   model selection inside folds; hard no-leakage assertion; within-data permutation
   chance baseline), then the gate, then a deployable final model.
4. **cluster_explore** — UDON program × subtype/dataset contingencies; flags
   batch-dominated programs.
5. **report** — `run_report.json` + `REPORT.md`.

## Layout (auto-detected)
- **local** working copy: modalities under `../data/`, labels flat under `../labels/`.
- **cluster** deposit: modalities at `<base>/RNA/…`, labels scattered
  (`Metadata/`, `RNA/clusters/`, `LSC-prediction/algorithm/`).
`dataio.resolve_paths` detects which from `base_dir`; the same code runs in both.

## Run it
Local (dev):
```
cd pipeline
python run.py --target subtype --strategy donor_kfold
python run.py --list-steps
```
Cluster (real runs, on the submit host e.g. bmiclusterp-head):
```
cd /data/salomonis-archive/LabFiles/Nicholas/AML-multimodal/pipeline
/usr/local/anaconda3-2020/bin/python submit.py configs        # list the matrix
/usr/local/anaconda3-2020/bin/python submit.py submit --dry-run
/usr/local/anaconda3-2020/bin/python submit.py submit         # fan out all configs
/usr/local/anaconda3-2020/bin/python submit.py status         # job + done + balacc + gate
```
Each config → one `bsub -L /bin/bash -n <slots> -M <mb> -W <wall> -R "span[hosts=1] rusage[mem=...]"`
job (no `-q` → the LSF **system default queue**, observed as `normal`; set `LSF_QUEUE`
or `--queue` to override). Jobs pin BLAS/sklearn threads to their slots. Add `--hooks agent`
to drive the decision seams with the LLM gateway. Outputs in `runs/<run_id>/`; check
`submit.py status` for per-config state (completed / errored / aborted / CRASHED? / CORRUPT).

## Validated baseline (cluster, sklearn 1.5.2)
subtype (5 driver classes: NPM1/Inv16/FLT3/TET2/TP53, n=97) from composition →
balanced accuracy **0.59**, permutation chance ceiling ~0.30, gate ACCEPT.

## Caveats baked into the design
- **Imputed ≠ independent:** GRN/metabolite/lipid/ADT are RNA-derived; use as
  interpretability/regularization, filtered by `heldout_spearman`. Not extra evidence.
- **Labels are the bottleneck:** feasibility flags survival (n≈19), WHO/FAB as not
  trainable. vital_status is empty.
- **Grouped CV is mandatory:** 12,255 pseudobulks / 387 specimens / 316 donors / 11
  cohorts → donor-grouped + leave-one-cohort-out, enforced by an assertion in `cv`.
- **Small-n sensitivity:** at n≈97 the point estimate moves ~0.06 across sklearn
  versions; the **cluster env is canonical**. The permutation p-value, not the raw
  number, is the trustworthy signal.

## Files
```
pipeline/
  run.py            one config, in-process (the LSF job body)
  submit.py         fan-out LSF submitter + status
  amlmm/
    dataio.py       layout-aware loader + modality aggregation
    cv.py           leakage-proof grouped/nested CV  (do not let agents replace)
    models.py       fold-safe model zoo
    hooks.py        DecisionHooks / AgentHooks — the agent seams
    step.py         StepSpec + registry + run_step
    targets.py      target/label definitions
    cluster.py      LSF bsub/bjobs backend (lab conventions)
    steps/          feasibility, assemble_features, classify, cluster_explore, report
  docs/AGENT_INTEGRATION.md
```
