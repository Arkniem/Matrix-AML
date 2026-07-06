#!/usr/bin/env python3
"""Orchestrator / CLI for the AML multimodal base pipeline.

Runs the deterministic step sequence today; designed so an agent can drive the
same steps later (swap --hooks, or call amlmm.step.run_step directly with chosen
params and branch on each StepResult). Examples:

  python run.py                                  # subtype, donor-grouped CV
  python run.py --target subtype --strategy leave_one_cohort_out
  python run.py --target subtype --blocks composition,ADT
  python run.py --list-steps
"""
from __future__ import annotations
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amlmm
from amlmm import step as stepmod
from amlmm import steps as steps_pkg   # registers steps
from amlmm.hooks import DecisionHooks, AgentHooks


def build_hooks(name):
    return AgentHooks() if name == "agent" else DecisionHooks()


def main(argv=None):
    ap = argparse.ArgumentParser(description="AML multimodal base pipeline (one config = one job)")
    ap.add_argument("--base", default=None, help="data base dir (auto-detects local vs cluster layout)")
    ap.add_argument("--out", default=None, help="output root (run dirs created under here)")
    ap.add_argument("--run-id", default="subtype_donorCV")
    ap.add_argument("--target", default="subtype")
    ap.add_argument("--strategy", default="donor_kfold",
                    choices=["donor_kfold", "leave_one_cohort_out"])
    ap.add_argument("--blocks", default=None, help="comma list, e.g. composition,ADT")
    ap.add_argument("--min-class-n", type=int, default=8)
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--hooks", default="deterministic", choices=["deterministic", "agent"])
    ap.add_argument("--steps", default=None, help="comma subset; default = full sequence")
    ap.add_argument("--list-steps", action="store_true")
    args = ap.parse_args(argv)

    if args.list_steps:
        for sp in stepmod.list_steps():
            print(f"{sp.name:18s} {sp.doc}")
            for k, v in sp.params_schema.items():
                print(f"    - {k} (default={v.get('default')}): {v.get('doc','')}")
        return 0

    _d = amlmm.Config()
    cfg = amlmm.Config(
        base_dir=args.base or _d.base_dir,
        out_dir=args.out or _d.out_dir,
        run_id=args.run_id,
    )
    ctx = amlmm.build_context(cfg, hooks=build_hooks(args.hooks))
    print(f"[ctx] layout={ctx.layout} {ctx.tables['coverage']}")
    print(f"[ctx] run dir: {ctx.run_dir}")

    # environment provenance (the headline metric depends on the sklearn build)
    def _ver(pkg):
        try:
            import importlib.metadata as m
            return m.version(pkg)
        except Exception:
            return "?"
    import platform
    ctx.set("provenance", {
        "python": platform.python_version(),
        "scikit-learn": _ver("scikit-learn"), "numpy": _ver("numpy"),
        "scipy": _ver("scipy"), "pandas": _ver("pandas"), "anndata": _ver("anndata"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "amlmm_njobs": os.environ.get("AMLMM_NJOBS"),
        "layout": ctx.layout, "hooks": args.hooks,
    })
    # clear stale artifacts so a re-run of this run_id can't blend old + new outputs
    for stale in ("run_report.json", "cv_result.json", "final_model.joblib", "REPORT.md"):
        fp = os.path.join(ctx.run_dir, stale)
        if os.path.exists(fp):
            os.remove(fp)

    blocks = [b.strip() for b in args.blocks.split(",")] if args.blocks else None
    params_by_step = {
        "feasibility": {"min_class_n": args.min_class_n},
        "assemble_features": {"target": args.target, "blocks": blocks,
                              "min_class_n": args.min_class_n},
        "classify": {"strategy": args.strategy, "outer_splits": args.outer_splits,
                     "inner_splits": args.inner_splits, "n_permutations": args.permutations},
        "cluster_explore": {},
        "report": {},
    }
    order = ([s.strip() for s in args.steps.split(",")] if args.steps
             else steps_pkg.DEFAULT_ORDER)

    had_error = False
    aborted = False
    for name in order:
        if aborted and name != "report":
            print(f"[--] {name} (skipped: run aborted by gate)")
            continue
        spec = stepmod.get(name)
        res = stepmod.run_step(spec, ctx, params_by_step.get(name, {}))
        flag = {"ok": "ok", "skipped": "--", "error": "!!"}.get(res.status, res.status)
        print(f"[{flag}] {name} ({res.seconds}s) "
              f"{ {k: v for k, v in res.metrics.items() if k not in ('per_fold',)} }")
        if res.status == "error":
            had_error = True
            print(f"     ERROR: {res.error}")
        if name == "classify":
            g = ctx.getart("gate")
            if g is not None and getattr(g, "action", None) == "abort":
                aborted = True
                print(f"     GATE ABORT: {g.reason} — skipping downstream analysis")

    status = "ERRORED" if had_error else ("aborted" if aborted else "completed")
    print(f"\nDone [{status}]. Outputs in: {ctx.run_dir}")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
