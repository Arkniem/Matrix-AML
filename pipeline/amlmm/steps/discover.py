"""`discover` step — the Discovery agent as a registered pipeline step.

Wraps `amlmm.discovery.run_discovery` so an orchestrator (or the `discover.py` CLI) can invoke it with
chosen knobs and read a StepResult. Self-registers on import. Reached only via `DISCOVERY_ORDER` /
`--steps discover`; the baseline `DEFAULT_ORDER` is untouched.
"""
from __future__ import annotations
from ..step import StepSpec, StepResult, register
from .. import discovery as D

# DiscoveryConfig knobs settable from params (the rest keep their dataclass defaults)
_CONFIG_KEYS = ("min_pseudobulks", "min_donors", "min_class_n", "min_donors_per_class", "alpha",
                "screen_permutations", "final_permutations", "screen_promote_alpha",
                "prefilter_features", "max_features", "feature_selector", "marker_k", "strategy")
_OUTPUTS = ("discovery_weights.tsv", "discovery_markers.tsv", "discovery_associations.tsv",
            "discovery_associations_index.json", "discovery_report.json", "DISCOVERY.md")


def _config_from_params(p):
    kw = {k: p[k] for k in _CONFIG_KEYS if p.get(k) is not None}
    return D.DiscoveryConfig(**kw)


def run(ctx, params) -> StepResult:
    res = StepResult(name="discover")
    cfg = _config_from_params(params)
    out = D.run_discovery(
        ctx, cfg,
        fields=(params.get("fields") or None),
        modalities=(params.get("modalities") or None),
        cell_states_top=int(params.get("cell_states_top", 40)),
        sample_level=bool(params.get("sample_level", True)),
        write=True, verbose=bool(params.get("verbose", True)))
    s = out["report"]["summary"]
    res.metrics = {"n_combos": s["n_combos"], "n_ok": s["n_ok"], "n_significant": s["n_significant"],
                   "n_skipped": s["n_skipped"], "n_error": s["n_error"],
                   "fields_predictable": s["fields_predictable"],
                   "n_markers": int(len(out["markers"])), "n_associations": int(len(out["associations"]))}
    for nm in _OUTPUTS:
        res.artifacts[nm] = ctx.path(nm)
    res.add_log("discovery: %s/%s significant combos; predictable=%s"
                % (s["n_significant"], s["n_ok"], s["fields_predictable"]))
    return res


SPEC = register(StepSpec(
    name="discover",
    run=run,
    doc=("Discovery agent: per-(field x modality x cell-state) permutation-calibrated weights + "
         "per-pseudobulk OOF associations + held-out-validated markers at pseudobulk resolution. "
         "Writes the three Discovery tables + association index + report."),
    params_schema={
        "fields": {"default": None, "doc": "subset of candidate_fields (None = all TARGETS + mutation flags)"},
        "modalities": {"default": None, "doc": "subset of modalities (None = all pseudobulk + sample-level)"},
        "cell_states_top": {"default": 40, "doc": "top-N cell-states by population (pseudobulk modalities)"},
        "sample_level": {"default": True, "doc": "include sample-level modalities (composition/cell-comm/LSC)"},
        "screen_permutations": {"default": 30, "doc": "pass-1 coarse permutation count (fail-fast)"},
        "final_permutations": {"default": 200, "doc": "pass-2 precise permutation count (survivors only)"},
        "screen_promote_alpha": {"default": 0.10, "doc": "promote screen->confirm iff screen perm_p <= this"},
        "prefilter_features": {"default": 800, "doc": "unsupervised top-variance cap applied before CV"},
        "max_features": {"default": 300, "doc": "in-fold supervised SelectKBest cap"},
        "feature_selector": {"default": "f_classif", "choices": ["f_classif", "mutual_info"],
                             "doc": "in-fold supervised selector"},
        "marker_k": {"default": 15, "doc": "top-K candidate markers harvested per combo"},
        "verbose": {"default": True, "doc": "per-combo logging"},
    },
    produces=["discovery_weights", "discovery_markers", "discovery_associations", "discovery_report"],
))
