#!/usr/bin/env python3
"""Fan-out LSF submitter — one bsub job per analysis configuration.

Each config = (target x CV strategy x feature blocks). Each is submitted as an
independent LSF job running `run.py` for that config, writing its own run dir
under <out>/<run_id>. This is the chosen execution model: per-config jobs,
embarrassingly parallel. Agents later read completed run dirs and submit more.

Usage (on the submit host, e.g. bmiclusterp-head):
  python submit.py configs                 # list the config matrix
  python submit.py submit --dry-run        # print the bsub commands
  python submit.py submit --limit 1        # submit one job (smoke test)
  python submit.py submit                  # submit the full matrix
  python submit.py status                  # job state + done/balacc/gate per config
"""
from __future__ import annotations
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from amlmm import cluster   # noqa: E402

DEFAULT_PYTHON = "/usr/local/anaconda3-2020/bin/python"
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "runs")   # AML-multimodal/runs


def default_matrix():
    """The fan-out grid. Edit / extend freely — each row becomes one LSF job."""
    cfgs = []
    # subtype (genetic driver): both CV strategies + two modality ablations
    cfgs.append({"target": "subtype", "strategy": "donor_kfold", "blocks": ["composition"]})
    cfgs.append({"target": "subtype", "strategy": "leave_one_cohort_out", "blocks": ["composition"]})
    cfgs.append({"target": "subtype", "strategy": "donor_kfold", "blocks": ["composition", "ADT"]})
    cfgs.append({"target": "subtype", "strategy": "donor_kfold", "blocks": ["composition", "GRN"]})
    # disease category (AML vs control vs ...)
    cfgs.append({"target": "disease_category", "strategy": "donor_kfold", "blocks": ["composition"]})
    # ELN risk
    cfgs.append({"target": "ELN_risk", "strategy": "donor_kfold", "blocks": ["composition"]})
    cfgs.append({"target": "ELN_risk", "strategy": "leave_one_cohort_out", "blocks": ["composition"]})
    return cfgs


def run_id_for(c):
    return f'{c["target"]}__{c["strategy"]}__{"+".join(c["blocks"])}'


def mem_for(blocks):
    # composition-only is tiny; modality blocks aggregate a big matrix -> size per modality
    big = {"RNA": 48000, "GRN": 32000, "Metabolite": 24000, "Lipid": 16000,
           "ADT": 12000, "cell-communication": 16000}
    mods = [b for b in blocks if b != "composition"]
    return max([big.get(b, 16000) for b in mods] + [8000]) if mods else 8000


def cmd_configs(args):
    for c in default_matrix():
        print(f'{run_id_for(c):48s} mem={mem_for(c["blocks"])}MB')


def cmd_submit(args):
    if not args.dry_run:
        cluster.require_bsub()
    out_root = args.out or DEFAULT_OUT
    os.makedirs(out_root, exist_ok=True)
    cfgs = default_matrix()
    if args.limit:
        cfgs = cfgs[:args.limit]
    rows = []
    for c in cfgs:
        rid = run_id_for(c)
        rdir = os.path.join(out_root, rid)
        os.makedirs(rdir, exist_ok=True)
        # clear prior result artifacts NOW so status during PEND/startup can't read a stale report
        if not args.dry_run:
            for stale in ("run_report.json", "cv_result.json", "final_model.joblib", "REPORT.md"):
                fp = os.path.join(rdir, stale)
                if os.path.exists(fp):
                    os.remove(fp)
        run_args = ["--run-id", rid, "--target", c["target"], "--strategy", c["strategy"],
                    "--blocks", ",".join(c["blocks"]), "--out", out_root,
                    "--permutations", str(args.permutations), "--hooks", args.hooks]
        if args.base:
            run_args += ["--base", args.base]
        script = cluster.write_job_script(os.path.join(rdir, "job.sh"), HERE,
                                          args.python, run_args, args.threads)
        res = cluster.submit(rid, script, rdir, threads=args.threads,
                             mem_mb=mem_for(c["blocks"]), wall=args.wall,
                             queue=args.queue, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[dry] {rid}\n      {res['cmd']}")
        else:
            tag = res.get("jobid") or f"FAIL(rc={res.get('rc')})"
            print(f"[{tag}] {rid}")
            if not res.get("jobid"):
                print(f"      stderr: {res.get('stderr')}")
        rows.append((rid, res.get("jobid") or "-", c["target"], c["strategy"],
                     "+".join(c["blocks"])))
    if not args.dry_run:
        man = os.path.join(out_root, "manifest.tsv")
        with open(man, "w") as f:
            f.write("run_id\tjobid\ttarget\tstrategy\tblocks\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        print(f"\nsubmitted {len(rows)} job(s); manifest: {man}")


def cmd_status(args):
    out_root = args.out or DEFAULT_OUT
    man = os.path.join(out_root, "manifest.tsv")
    if not os.path.exists(man):
        print(f"no manifest at {man}")
        return
    print(f'{"run_id":46s} {"job":>7s} {"lsf":>9s} {"state":>11s} {"balacc":>7s} {"permP":>6s} {"gate":>7s}')
    for line in open(man).read().splitlines()[1:]:
        rid, jobid, target, strat, blocks = line.split("\t")
        stat = cluster.job_stat_by_id(jobid) if jobid not in ("-", "") else cluster.job_stat(rid)
        rep = os.path.join(out_root, rid, "run_report.json")
        state, ba, pv, gate = "PENDING", "", "", ""
        if os.path.exists(rep):
            try:
                d = json.load(open(rep))
                state = d.get("run_status", "completed")
                cv = d.get("cv") or {}
                ba = str(cv.get("balanced_accuracy", "") or "")
                pv = str(cv.get("permutation_pvalue", "") or "")[:5]
                g = d.get("gate") or {}
                gate = ("ACCEPT" if g.get("accept") else "REJECT") if g else ""
            except Exception:
                state = "CORRUPT"          # partial/truncated report (e.g. job killed mid-write)
        elif stat == "GONE":
            state = "CRASHED?"             # left the queue but produced no report
        elif stat in ("UNKNOWN", "AMBIGUOUS"):
            state = stat
        print(f"{rid:46s} {jobid:>7s} {stat:>9s} {state:>11s} {ba:>7s} {pv:>6s} {gate:>7s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fan-out LSF submitter for the AML multimodal pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="submit per-config jobs")
    s.set_defaults(fn=cmd_submit)
    s.add_argument("--queue", default=os.environ.get("LSF_QUEUE"))
    s.add_argument("--python", default=DEFAULT_PYTHON)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--wall", default="120:00")
    s.add_argument("--permutations", type=int, default=200)
    s.add_argument("--hooks", default="deterministic", choices=["deterministic", "agent"],
                   help="agent = LLM-driven decision seams (nemotron via the gateway)")
    s.add_argument("--base", default=None, help="data base dir (default: auto)")
    s.add_argument("--out", default=None)
    s.add_argument("--limit", type=int, default=None, help="submit only the first N configs")
    s.add_argument("--dry-run", action="store_true")

    st = sub.add_parser("status", help="report job + completion state")
    st.set_defaults(fn=cmd_status)
    st.add_argument("--out", default=None)

    cf = sub.add_parser("configs", help="list the config matrix")
    cf.set_defaults(fn=cmd_configs)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
