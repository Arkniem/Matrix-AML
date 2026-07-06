"""Curated knowledge / rules layer — versioned, auditable TSVs.

biomarker_drug.tsv and validation_rules.tsv (in amlmm/knowledge/) make the arbiter's
therapy and validation recommendations reproducible: the LLM may only order/narrate
within this fixed candidate set, never invent. Seeded from genetics.TARGETABLE
(`seed_check` asserts coverage). Read with the repo's TSV convention.
"""
from __future__ import annotations
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(_HERE, "knowledge")
_ORDER = {"guideline": 0, "trial": 1, "preclinical": 2, "heuristic": 3}


class KnowledgeBase:
    def __init__(self, biomarker_drug: pd.DataFrame, validation_rules: pd.DataFrame, version: str):
        self.biomarker_drug = biomarker_drug
        self.validation_rules = validation_rules
        self.version = version

    def therapies_for(self, biomarkers) -> list:
        """Ranked candidate therapies for a set of present biomarkers (by evidence level)."""
        if self.biomarker_drug.empty:
            return []
        bset = {str(b) for b in biomarkers}
        df = self.biomarker_drug[self.biomarker_drug["biomarker"].isin(bset)].copy()
        df["_o"] = df["evidence_level"].map(lambda x: _ORDER.get(str(x), 9))
        # explicit tiebreak + stable sort so equal-evidence rows order identically across
        # pandas versions / larger driver sets (the default quicksort is not stable).
        df = df.sort_values(["_o", "biomarker", "drug"], kind="mergesort")
        return [{"biomarker": r["biomarker"], "drug": r["drug"],
                 "evidence_level": r["evidence_level"], "source": r["source"]}
                for _, r in df.iterrows()]

    def validations_for(self, claim_types) -> list:
        if self.validation_rules.empty:
            return []
        cset = {str(c) for c in claim_types}
        df = self.validation_rules[self.validation_rules["claim_type"].isin(cset)]
        return [{"claim": r["claim_type"], "validation": r["validation"], "source": r["source"]}
                for _, r in df.iterrows()]


def load_knowledge(dirpath: str | None = None) -> KnowledgeBase:
    d = dirpath or KB_DIR

    def rd(fname, required):
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            return pd.DataFrame(columns=sorted(required))
        df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        # validate the schema up front: a missing/misspelled header otherwise
        # surfaces as an uncaught KeyError inside the arbiter for every patient.
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"knowledge/{fname} missing required column(s) "
                             f"{sorted(missing)} (found {list(df.columns)})")
        return df

    bd = rd("biomarker_drug.tsv", {"biomarker", "drug", "evidence_level", "source"})
    # drop ragged/incomplete rows so a truncated edit never emits a therapy with
    # no drug or no provenance.
    if not bd.empty:
        bd = bd[(bd["biomarker"].str.strip() != "")
                & (bd["drug"].str.strip() != "")].reset_index(drop=True)
    vr = rd("validation_rules.tsv", {"claim_type", "validation", "source"})

    version = "?"
    vp = os.path.join(d, "VERSION")
    if os.path.exists(vp):
        with open(vp) as f:
            version = f.read().strip()
    return KnowledgeBase(bd, vr, version)


def seed_check(kb: KnowledgeBase) -> set:
    """Return TARGETABLE genes NOT covered by biomarker_drug (should be empty)."""
    from .genetics import TARGETABLE
    # mirror panel._patient_genetic: strip BOTH mut_ and cyto_ prefixes so a future
    # cytogenetic targetable (e.g. cyto_inv16) is not falsely reported as a gap.
    genes = {g.replace("mut_", "").replace("cyto_", "") for g in TARGETABLE}
    covered = set(kb.biomarker_drug["biomarker"]) if not kb.biomarker_drug.empty else set()
    return genes - covered
