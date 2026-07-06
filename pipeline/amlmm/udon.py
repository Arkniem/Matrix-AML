"""UDON program associations (SATAY-UDON style) — the cell-state witness's grounding.

Cross-tabulates the conserved UDON RNA programs against genetic subtype and against
dataset, surfacing which programs associate with which subtype (phenotype-genotype
links) and which programs are dataset-dominated (batch, to distrust). This is the
discovery evidence the cell-state/UDON agent reasons from.
"""
from __future__ import annotations
import pandas as pd

_GENERIC = ("AML", "Control", "Pediatric-AML", "nan", "0", "")


def udon_associations(ctx) -> dict:
    prog = ctx.tables.get("udon_programs")
    if prog is None or not {"Dataset", "Sample", "final_program"}.issubset(prog.columns):
        return {"available": False, "reason": "udon_programs missing or wrong schema"}
    s = ctx.tables["samples"]
    m = ctx.hooks.canonical_label_map(s["annotation"].tolist())
    prog = prog.copy()
    prog["sample_key"] = [f"{d}::{x}" for d, x in zip(prog["Dataset"], prog["Sample"])]
    prog["subtype"] = prog["sample_key"].map(s["annotation"]).map(m)
    prog = prog.dropna(subset=["final_program"])

    ct_sub = pd.crosstab(prog["final_program"], prog["subtype"])
    ct_ds = pd.crosstab(prog["final_program"], prog["Dataset"])
    ds_frac = (ct_ds.max(axis=1) / ct_ds.sum(axis=1)).fillna(0)

    assoc = {}
    drop = [c for c in _GENERIC if c in ct_sub.columns]
    specific = ct_sub.drop(columns=drop, errors="ignore")
    for p in specific.index:
        row = specific.loc[p]
        if row.sum() > 0:
            assoc[str(p)] = {"top_subtype": str(row.idxmax()), "n": int(row.max()),
                             "dataset_dominated": bool(ds_frac.get(p, 0) > 0.8)}
    # programs ranked by how specifically they mark a (non-generic) subtype
    ranked = sorted(assoc.items(), key=lambda kv: -kv[1]["n"])
    return {
        "available": True,
        "n_programs": int(prog["final_program"].nunique()),
        "n_pseudobulks": int(len(prog)),
        "program_subtype_top": dict(ranked[:12]),
        "n_batch_dominated": int((ds_frac > 0.8).sum()),
        "batch_dominated_programs": [str(p) for p in ds_frac.index[ds_frac > 0.8][:8]],
    }
