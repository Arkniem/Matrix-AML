#!/usr/bin/env python3
"""Fan-out LSF submitter for the Discovery agent — one bsub job PER metadata FIELD.

Each job runs `discover.py` for one field over the requested modalities x top-N cell-states (200 perms),
writing its own run dir <out>/disc__<field>/. Embarrassingly parallel (fields are independent). A final
`merge` job stitches the per-field outputs into one combined sweep (no CV re-run). Discovery runs SERIAL
inside each job (AMLMM_NJOBS=1) per the convergence/CPU-limit finding; parallelism is across jobs.

SELF-CONTAINED + 3.6.8-safe (no amlmm import, no `from __future__`): the login node's stock
/usr/bin/python3 cannot import amlmm (pandas/numpy 2.4.6 needs a compute node's GLIBC), so this submitter
inlines the tiny LSF logic and only orchestrates. The WORK runs on compute nodes via the anaconda python.

Usage (on the submit host, bmiclusterp-head):
  /usr/bin/python3 submit_discovery.py configs
  /usr/bin/python3 submit_discovery.py submit --dry-run
  /usr/bin/python3 submit_discovery.py submit --limit 1      # one field (smoke)
  /usr/bin/python3 submit_discovery.py submit                # full per-field fan-out
  /usr/bin/python3 submit_discovery.py status
  /usr/bin/python3 submit_discovery.py merge                 # after all Done -> disc_sweep/
"""
import os
import re
import sys
import json
import shlex
import argparse
import subprocess
from shutil import which

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PYTHON = "/usr/local/anaconda3-2020/bin/python"
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "runs")          # AML-multimodal/runs
METADATA_FIELDS = ["subtype", "ELN_risk", "is_pediatric", "disease_category"]   # plan's full scope
DEFAULT_MODALITIES = "composition,RNA,ADT,Metabolite"
_JOBID = re.compile(r"Job <(\d+)>")


# --------------------------------------------------------------------------- inlined LSF helpers (stdlib)
def require_bsub():
    if which("bsub") is None:
        raise SystemExit("ERROR: 'bsub' not found — run on the LSF submit host (bmiclusterp-head).")


def lsf_submit(name, script_path, log_dir, threads, mem_mb, wall, queue, dry_run):
    out = os.path.join(log_dir, name + ".lsf.out")
    err = os.path.join(log_dir, name + ".lsf.err")
    argv = ["bsub", "-L", "/bin/bash"]
    if queue:
        argv += ["-q", queue]
    argv += ["-J", name, "-n", str(int(threads)), "-W", str(wall), "-M", str(int(mem_mb)),
             "-R", "span[hosts=1] rusage[mem=%d]" % int(mem_mb), "-o", out, "-e", err,
             "bash", script_path]
    cmd = " ".join(shlex.quote(a) for a in argv)
    if dry_run:
        return {"cmd": cmd, "jobid": None}
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    m = _JOBID.search(p.stdout or "")
    return {"cmd": cmd, "jobid": (m.group(1) if m else None),
            "stderr": (p.stderr or "").strip(), "rc": p.returncode}


def job_stat_by_id(jobid):
    """Fail-closed: bjobs error -> UNKNOWN (never falsely DONE); 'GONE' = left the queue."""
    if jobid in ("-", "", None):
        return "UNKNOWN"
    p = subprocess.run(["bjobs", "-noheader", "-o", "stat", str(jobid)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode != 0:
        return "UNKNOWN"
    toks = (p.stdout or "").split()
    return toks[-1] if toks else "GONE"


# --------------------------------------------------------------------------- the field-job matrix
def default_matrix(modalities, with_mutations=True):
    jobs = [{"field": f, "mutations_only": False} for f in METADATA_FIELDS]
    if with_mutations:
        jobs.append({"field": "mutations", "mutations_only": True})
    for j in jobs:
        j["modalities"] = modalities
    return jobs


def run_id_for(j):
    return "disc__" + j["field"]


def mem_for(modalities):
    mods = [m.strip() for m in modalities.split(",") if m.strip()]
    return 48000 if "RNA" in mods else 24000          # RNA loads a full ~12k x 36k matrix non-backed


def write_job(path, python_bin, run_args, threads):
    """Local job-script writer (does NOT touch the baseline cluster.write_job_script). Pins BLAS threads
    to the slots but forces AMLMM_NJOBS=1 (sklearn serial — the CPU-limit fix)."""
    t = str(int(threads))
    lines = [
        "#!/bin/bash", "set -euo pipefail",
        "export OMP_NUM_THREADS=" + t, "export OPENBLAS_NUM_THREADS=" + t,
        "export MKL_NUM_THREADS=" + t, "export NUMEXPR_NUM_THREADS=" + t,
        "export AMLMM_NJOBS=1",
        "cd " + shlex.quote(HERE),
        shlex.quote(python_bin) + " -u discover.py " + " ".join(shlex.quote(a) for a in run_args),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def common_args(args, j, out_root):
    rid = run_id_for(j)
    a = ["--run-id", rid, "--modalities", j["modalities"], "--out", out_root,
         "--cell-states-top", str(args.cell_states_top),
         "--screen-permutations", str(args.screen_permutations),
         "--final-permutations", str(args.final_permutations)]
    if j["mutations_only"]:
        a.append("--mutations-only")
    else:
        a += ["--fields", j["field"]]
    if args.base:
        a += ["--base", args.base]
    return rid, a


# --------------------------------------------------------------------------- subcommands
def cmd_configs(args):
    for j in default_matrix(args.modalities, not args.no_mutations):
        f = "<mutations>" if j["mutations_only"] else j["field"]
        print("%-28s fields=%-18s mods=%s mem=%dMB" % (run_id_for(j), f, j["modalities"],
                                                       mem_for(j["modalities"])))


def cmd_submit(args):
    if not args.dry_run:
        require_bsub()
    out_root = args.out or DEFAULT_OUT
    os.makedirs(out_root, exist_ok=True)
    jobs = default_matrix(args.modalities, not args.no_mutations)
    if args.limit:
        jobs = jobs[:args.limit]
    rows = []
    for j in jobs:
        rid, run_args = common_args(args, j, out_root)
        rdir = os.path.join(out_root, rid)
        os.makedirs(rdir, exist_ok=True)
        if not args.dry_run:
            for stale in ("discovery_weights.tsv", "discovery_report.json", "DISCOVERY.md"):
                fp = os.path.join(rdir, stale)
                if os.path.exists(fp):
                    os.remove(fp)
        script = write_job(os.path.join(rdir, "job.sh"), args.python, run_args, args.threads)
        res = lsf_submit(rid, script, rdir, args.threads, mem_for(j["modalities"]),
                         args.wall, args.queue, args.dry_run)
        if args.dry_run:
            print("[dry] %s\n      %s" % (rid, res["cmd"]))
        else:
            print("[%s] %s" % (res.get("jobid") or ("FAIL(rc=%s)" % res.get("rc")), rid))
            if not res.get("jobid"):
                print("      stderr: %s" % res.get("stderr"))
        rows.append((rid, res.get("jobid") or "-", j["modalities"]))
    if not args.dry_run:
        man = os.path.join(out_root, "discovery_manifest.tsv")
        with open(man, "w") as f:
            f.write("run_id\tjobid\tmodalities\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        print("\nsubmitted %d job(s); manifest: %s" % (len(rows), man))


def cmd_status(args):
    out_root = args.out or DEFAULT_OUT
    man = os.path.join(out_root, "discovery_manifest.tsv")
    if not os.path.exists(man):
        print("no manifest at %s" % man)
        return
    print("%-28s %8s %9s %10s %7s %4s  %s" % ("run_id", "job", "lsf", "state", "combos", "sig", "predictable"))
    for line in open(man).read().splitlines()[1:]:
        parts = line.split("\t")
        rid, jobid = parts[0], parts[1]
        stat = job_stat_by_id(jobid)
        rep = os.path.join(out_root, rid, "discovery_report.json")
        state, ncombos, nsig, pred = "PENDING", "", "", ""
        if os.path.exists(rep):
            try:
                d = json.load(open(rep))
                s = d.get("summary", {})
                state = "done"
                ncombos, nsig = str(s.get("n_combos", "")), str(s.get("n_significant", ""))
                pred = ",".join(s.get("fields_predictable", []))[:40]
            except Exception:
                state = "CORRUPT"
        elif stat == "GONE":
            state = "CRASHED?"
        elif stat in ("UNKNOWN",):
            state = stat
        print("%-28s %8s %9s %10s %7s %4s  %s" % (rid, jobid, stat, state, ncombos, nsig, pred))


def cmd_merge(args):
    if not args.dry_run:
        require_bsub()
    out_root = args.out or DEFAULT_OUT
    rid = args.merge_run_id
    rdir = os.path.join(out_root, rid)
    os.makedirs(rdir, exist_ok=True)
    run_args = ["--run-id", rid, "--out", out_root, "--merge-from", "auto"]
    if args.base:
        run_args += ["--base", args.base]
    script = write_job(os.path.join(rdir, "job.sh"), args.python, run_args, 1)
    res = lsf_submit(rid, script, rdir, 1, 16000, "4:00", args.queue, args.dry_run)
    if args.dry_run:
        print("[dry] %s\n      %s" % (rid, res["cmd"]))
    else:
        print("[%s] %s (merge); outputs -> %s" % (res.get("jobid") or "FAIL", rid, rdir))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fan-out LSF submitter for the Discovery agent")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--queue", default=os.environ.get("LSF_QUEUE"))
        p.add_argument("--python", default=DEFAULT_PYTHON)
        p.add_argument("--out", default=None)
        p.add_argument("--base", default=None)
        p.add_argument("--modalities", default=DEFAULT_MODALITIES)
        p.add_argument("--no-mutations", action="store_true", help="skip the combined mutations job")

    s = sub.add_parser("submit", help="submit per-field jobs")
    s.set_defaults(fn=cmd_submit)
    common(s)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--wall", default="48:00")
    s.add_argument("--cell-states-top", type=int, default=40)
    s.add_argument("--screen-permutations", type=int, default=30)
    s.add_argument("--final-permutations", type=int, default=200)
    s.add_argument("--limit", type=int, default=None, help="submit only the first N field-jobs")
    s.add_argument("--dry-run", action="store_true")

    st = sub.add_parser("status", help="report per-field job + completion state")
    st.set_defaults(fn=cmd_status)
    st.add_argument("--out", default=None)

    cf = sub.add_parser("configs", help="list the field-job matrix")
    cf.set_defaults(fn=cmd_configs)
    common(cf)

    mg = sub.add_parser("merge", help="merge finished disc__* dirs into one combined sweep")
    mg.set_defaults(fn=cmd_merge)
    common(mg)
    mg.add_argument("--merge-run-id", default="disc_sweep")
    mg.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return
    args.fn(args)


if __name__ == "__main__":
    main()
