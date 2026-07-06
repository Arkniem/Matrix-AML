"""Derive a per-sample mutation / cytogenetic matrix from the sparse metadata.

Sources: the obs-level driver `annotation` (clean driver calls) + free-text
`karyotype` (cytogenetic events) + `ELN_risk`. Honest about coverage: a flag is
1 when there is positive evidence of the lesion and NaN when it is simply not
mentioned -- we do NOT assert absence from sparse free text. This feeds the
genetic witness (risk context + targetable drivers), not a subtype predictor.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# canonical-annotation driver label -> mutation/cytogenetic flag
ANNOT_TO_FLAG = {
    "NPM1": "mut_NPM1", "NPM1c": "mut_NPM1", "FLT3": "mut_FLT3", "FLT3-ITD": "mut_FLT3",
    "TP53": "mut_TP53", "TP53 U2AF1": "mut_TP53", "IDH1": "mut_IDH1", "IDH2": "mut_IDH2",
    "DNMT3A": "mut_DNMT3A", "TET2": "mut_TET2", "ASXL1": "mut_ASXL1", "RUNX1": "mut_RUNX1",
    "SRSF2": "mut_SRSF2", "SF3B1": "mut_SF3B1", "U2AF1": "mut_U2AF1", "KRAS": "mut_KRAS",
    "NRAS": "mut_NRAS", "CEBPA": "mut_CEBPA", "WT1": "mut_WT1", "CBL": "mut_CBL",
    "CKIT": "mut_KIT", "KIT": "mut_KIT", "CSF3R": "mut_CSF3R",
    "Inv16": "cyto_inv16", "DEL(7)": "cyto_del7",
}

TARGETABLE = {
    "mut_FLT3": "FLT3 inhibitor (midostaurin / gilteritinib)",
    "mut_IDH1": "IDH1 inhibitor (ivosidenib)",
    "mut_IDH2": "IDH2 inhibitor (enasidenib)",
    "mut_NPM1": "menin inhibitor (NPM1/KMT2A-rearranged context)",
    "mut_KIT": "KIT inhibitor (avapritinib; CBF-AML context)",
}

KARYO_PATTERNS = {
    "cyto_inv16": r"inv\(16\)|t\(16;16\)|cbf",
    "cyto_t8_21": r"t\(8;21\)|runx1-runx1t1|aml1-eto",
    "cyto_t15_17": r"t\(15;17\)|pml-rara",
    "cyto_kmt2a": r"11q23|kmt2a|\bmll\b|t\(\d+;11\)",
    "cyto_del7": r"-7\b|del\(7|monosomy ?7|7q-",
    "cyto_del5": r"-5\b|del\(5|5q-",
    "cyto_trisomy8": r"\+8\b|trisomy ?8",
}


def _free(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _complexity(t):
    t = t.lower()
    if "complex" in t:
        return 1.0
    n = len(re.findall(r"t\(|del\(|inv\(|add\(|der\(|dup\(|\+\d|-\d", t))
    return 1.0 if n >= 3 else np.nan


def _overlay_explicit_calls(M):
    """Overlay the explicit per-mutation matrix (pipeline/mutation_matrix_explicit.tsv, built from the
    metadata sheet's FULL co-mutation profile) onto the annotation/karyotype-derived flags. The single
    `annotation` label only records each sample's primary driver, so secondary/co-occurring mutations
    (IDH2, DNMT3A, NRAS, ...) are otherwise invisible. Explicit calls (present=1 AND absent=0) take
    precedence wherever the sheet gives a value; samples/flags absent from the sheet keep the derived
    value. NaN in the sheet never overwrites. Set AMLMM_MUTATIONS=<path>, or "" to disable."""
    import os
    p = os.environ.get("AMLMM_MUTATIONS",
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "mutation_matrix_explicit.tsv"))
    if not p or not os.path.exists(p):
        return M
    try:
        E = pd.read_csv(p, sep="\t", index_col=0)
    except Exception:
        return M
    for col in E.columns:
        if not str(col).startswith(("mut_", "cyto_")):
            continue
        if col not in M.columns:
            M[col] = np.nan
        v = pd.to_numeric(E[col], errors="coerce").reindex(M.index)
        idx = v.index[v.notna()]
        M.loc[idx, col] = v.loc[idx]
    return M


def build_mutation_matrix(ctx) -> pd.DataFrame:
    s = ctx.tables["samples"]
    flags = sorted(set(ANNOT_TO_FLAG.values()) | set(KARYO_PATTERNS) | {"cyto_complex"})
    M = pd.DataFrame(index=s.index, columns=flags, dtype=float)

    ann = s["annotation"].astype(str)
    for label, flag in ANNOT_TO_FLAG.items():
        M.loc[ann.eq(label), flag] = 1.0

    kar = (s["karyotype"].map(_free).str.lower() if "karyotype" in s.columns
           else pd.Series("", index=s.index))
    for flag, pat in KARYO_PATTERNS.items():
        M.loc[kar.str.contains(pat, regex=True, na=False), flag] = 1.0
    M["cyto_complex"] = kar.map(_complexity)

    M = _overlay_explicit_calls(M)     # full co-mutation profile from the metadata sheet (takes precedence)

    if "ELN_risk" in s.columns:
        M["ELN_risk"] = s["ELN_risk"]
    M["has_genetic_data"] = (M.filter(regex="^(mut_|cyto_)").notna().any(axis=1)
                             | (M["ELN_risk"].notna() if "ELN_risk" in M.columns else False))
    ctx.tables["mutations"] = M
    return M


def summarize_genetics(ctx, sample_keys=None) -> dict:
    M = ctx.tables.get("mutations")
    if M is None:
        M = build_mutation_matrix(ctx)
    if sample_keys is not None:
        M = M.loc[[k for k in sample_keys if k in M.index]]
    flags = M.filter(regex="^(mut_|cyto_)")
    prev = {c: int(flags[c].fillna(0).sum()) for c in flags.columns}
    prev = {k.replace("mut_", "").replace("cyto_", ""): v
            for k, v in sorted(prev.items(), key=lambda x: -x[1]) if v > 0}
    targetable = {c.replace("mut_", ""): TARGETABLE[c] for c in TARGETABLE
                  if c in M.columns and M[c].fillna(0).sum() > 0}
    eln = (M["ELN_risk"].dropna().value_counts().to_dict() if "ELN_risk" in M.columns else {})
    return {
        "n_samples": int(len(M)),
        "n_with_genetic_data": int(M["has_genetic_data"].sum()) if "has_genetic_data" in M else 0,
        "mutation_prevalence": prev,
        "targetable_present": targetable,
        "eln_distribution": {str(k): int(v) for k, v in eln.items()},
    }
