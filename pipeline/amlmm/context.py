"""Config + Context — paths, loaded tables, the artifact store, and the hooks.

Context is what every step receives. It carries the loaded data (ctx.tables),
lazy modality access (ctx.open_modality), an artifact store for passing results
between steps (ctx.set / ctx.getart), the decision hooks (ctx.hooks), and IO
helpers that write into a per-run output directory.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field

from .hooks import DecisionHooks
from . import dataio

_HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(_HERE)
ROOT = os.path.dirname(PIPELINE_DIR)               # .../AML-multimodal


@dataclass
class Config:
    # base_dir holds the data; layout (local data/+labels/ vs cluster scattered)
    # is auto-detected by dataio.resolve_paths. Defaults work both places.
    base_dir: str = ROOT
    out_dir: str = os.path.join(ROOT, "runs")
    run_id: str = "run"
    extra: dict = field(default_factory=dict)


class Context:
    def __init__(self, config: Config, hooks: DecisionHooks | None = None):
        self.config = config
        self.hooks = hooks or DecisionHooks()
        self.layout: str | None = None
        self.knowledge = None          # curated KnowledgeBase (set by build_context)
        self.ledger = None             # shared evidence ledger (set per patient/cohort run)
        self.holdout: set = set()      # sample_keys excluded from Discovery training (a held-out test set)
        self.tables: dict = {}
        self.artifacts: dict = {}
        self.results: list = []
        self._modality_paths: dict = {}
        self.run_dir = os.path.join(config.out_dir, config.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    # --- artifact store (inter-step plumbing) ---
    def set(self, key, value):
        self.artifacts[key] = value

    def getart(self, key, default=None):
        return self.artifacts.get(key, default)

    def record(self, result):
        self.results.append(result)

    # --- modalities (lazy, backed) ---
    def open_modality(self, name):
        import anndata as ad
        return ad.read_h5ad(self._modality_paths[name], backed="r")

    def modalities(self):
        return list(self._modality_paths)

    # --- IO helpers (per-run output dir) ---
    def path(self, *parts):
        return os.path.join(self.run_dir, *parts)

    def save_table(self, df, name, index=True):
        fp = self.path(name)
        df.to_csv(fp, sep="\t", index=index)
        return fp

    def save_json(self, obj, name):
        # atomic: write to a temp file then rename, so an interrupted job never
        # leaves a truncated run_report.json that the status check reads as success.
        fp = self.path(name)
        tmp = fp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, fp)
        return fp


def build_context(config: Config | None = None, hooks: DecisionHooks | None = None) -> Context:
    ctx = Context(config or Config(), hooks=hooks)
    dataio.load_into(ctx)
    try:
        from . import knowledge
        ctx.knowledge = knowledge.load_knowledge()
    except Exception:
        ctx.knowledge = None
    # optional held-out test set: sample_keys in pipeline/holdout_samples.txt are excluded from
    # Discovery TRAINING (labels masked) so they can be predicted as an honest external test. Absent
    # file -> empty set -> baseline behavior unchanged. Override via env AMLMM_HOLDOUT (path) or "" to disable.
    ho_path = os.environ.get("AMLMM_HOLDOUT", os.path.join(PIPELINE_DIR, "holdout_samples.txt"))
    if ho_path and os.path.exists(ho_path):
        with open(ho_path, encoding="utf-8") as f:
            ctx.holdout = {ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")}
    return ctx
