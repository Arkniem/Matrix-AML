"""Step: cluster_explore — unsupervised view of UDON RNA programs vs labels.

Cross-tabulates each UDON program against (a) the canonical genetic subtype and
(b) the dataset of origin, then flags programs that are dataset-dominated (a
batch-confounding red flag). No modeling; this is the "are the clusters biology
or batch?" sanity check that should run before trusting any program as a feature.
"""
from __future__ import annotations
import pandas as pd

from ..step import StepSpec, StepResult, register


def run(ctx, params) -> StepResult:
    prog = ctx.tables["udon_programs"].copy()
    need = {"Dataset", "Sample", "final_program"}
    if not need.issubset(prog.columns):
        return StepResult(name="cluster_explore", status="skipped",
                          metrics={"reason": f"udon_programs missing columns "
                                             f"{sorted(need - set(prog.columns))}"})
    s = ctx.tables["samples"]
    m = ctx.hooks.canonical_label_map(s["annotation"].tolist())
    prog["sample_key"] = [f"{d}::{x}" for d, x in zip(prog["Dataset"], prog["Sample"])]
    prog["subtype"] = prog["sample_key"].map(s["annotation"]).map(m)
    prog = prog.dropna(subset=["final_program"])

    ct_sub = pd.crosstab(prog["final_program"], prog["subtype"])
    ct_ds = pd.crosstab(prog["final_program"], prog["Dataset"])
    ds_frac = (ct_ds.max(axis=1) / ct_ds.sum(axis=1)).sort_values(ascending=False)

    fp1 = ctx.save_table(ct_sub, "program_by_subtype.tsv")
    fp2 = ctx.save_table(ct_ds, "program_by_dataset.tsv")

    res = StepResult(name="cluster_explore",
                     artifacts={"program_by_subtype.tsv": fp1, "program_by_dataset.tsv": fp2})
    res.metrics = {
        "n_programs": int(prog["final_program"].nunique()),
        "n_pseudobulks_assigned": int(len(prog)),
        "programs_dataset_dominated_gt80pct": int((ds_frac > 0.8).sum()),
        "example_batch_dominated": list(ds_frac.index[ds_frac > 0.8][:5]),
        "max_dataset_fraction_median": round(float(ds_frac.median()), 3),
    }
    return res


STEP = register(StepSpec(
    name="cluster_explore",
    run=run,
    doc="UDON program x subtype and program x dataset contingencies; flag batch-driven programs.",
    params_schema={},
))
