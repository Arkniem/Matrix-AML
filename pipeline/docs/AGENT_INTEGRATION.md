# Agent integration guide

This pipeline is built so LLM agents can drive it **without touching the
deterministic core**. There are two layers:

- **Deterministic core** — `dataio`, `cv` (the leakage-proof grouped/nested CV),
  `models`, `step` mechanics. Reproducible, auditable, identical every run. Agents
  never reimplement this; especially not `cv` (the project's credibility lives there).
- **Decision hooks** (`amlmm/hooks.py`) — the *seams*. The pipeline calls these at
  every judgment point. Defaults are auditable (`DecisionHooks`); swap in `AgentHooks`
  to let an agent decide, returning the same types. Nothing else changes.

## The five seams

| Hook method | Decision | Default | What an agent adds |
|---|---|---|---|
| `canonical_label_map(annotations)` | how to merge raw driver labels | NPM1c→NPM1, FLT3-ITD→FLT3, … | genetics judgment (split/merge subtypes) |
| `select_feature_blocks(target, available, feasibility)` | which modalities to use | `composition` | run/choose an ablation; add modality blocks |
| `choose_models(target, feasibility)` | model panel | `rf`, `logreg` | pick models per task shape |
| `gate_result(cv_result)` | accept / rerun / abort | beat permutation p95 | **adversarial skeptic**: hunt for leakage, demand honest CV |
| `synthesize_report(run_report)` | the write-up | deterministic template | clinician-facing narrative with caveats |

`--hooks agent` is **live**: `AgentHooks` (`amlmm/hooks.py`) routes `gate_result`,
`select_feature_blocks`, `choose_models`, and `synthesize_report` to the LLM gateway
(default `nemotron-3-super` via `amlmm/llm.py`; override with `AMLMM_LLM_BASE_URL` /
`AMLMM_LLM_API_KEY` / `AMLMM_LLM_MODEL`). Every seam validates the model's reply and
falls back to the deterministic default on any error, so a gateway outage degrades
gracefully rather than crashing. `canonical_label_map` stays deterministic by design.
To customize, subclass `DecisionHooks`/`AgentHooks` and pass it to
`build_context(cfg, hooks=...)`. Adopt agents one seam at a time; the skeptic
(`gate_result`) is the highest-value seam.

## Two levels at which an agent intervenes

**Within a run (in-process):** the hooks fire between steps inside one job — the
read→decide→gate loop, with the deterministic step doing the math. `gate_result`
returning `action="abort"` **is acted on** (run.py skips the remaining analysis steps
and stamps `run_status="aborted"`); `action="rerun"` is recorded but realized at the
job level (the agent resubmits a new config), not as an in-process retry. Note
`select_feature_blocks` is the in-run seam used only when a job omits `--blocks`;
since `submit.py` passes explicit `--blocks`, across-runs block/ablation choice lives
in the agent-generated config matrix, not this hook.

**Across runs (the job level):** the chosen execution model is **per-config LSF jobs**
(`submit.py`). An agent loop here is:

1. submit a batch of configs (target × strategy × feature blocks) → `submit.py`.
2. read each finished `runs/<run_id>/run_report.json` (the structured *readout*).
3. decide the next batch (e.g. "subtype is above chance on composition — try
   `composition+ADT`; ELN_risk failed leave-one-cohort-out — it's confounded, drop it").
4. submit, repeat until a budget/criterion is met.

Each step is also a registered `StepSpec` (`amlmm.step.REGISTRY`) carrying a
`params_schema` (the knobs) and `consumes`/`produces` (the artifact contract), so an
agent can enumerate steps, read their schemas, and invoke `run_step(spec, ctx, params)`
directly for finer control than the job level.

## The one rule

The agent owns **which** config/model/features and **whether to trust** a result.
It does **not** own the CV splitting, the leakage assertion, or the metric
computation — those are fixed code it calls and interprets. This is exactly the
discipline that keeps an enthusiastic panel of agents from reporting inflated
numbers (cf. the LSC bake-off's honest 0.85→0.59 correction).
