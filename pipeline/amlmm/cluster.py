"""LSF backend — submit a per-config job and track it, matching the lab's
SpliceScout conventions:

  bsub -L /bin/bash [-q QUEUE] -J <name> -n <slots> -W <wall> -M <mem_mb> -o .. -e ..
  * queue optional (default cluster queue when unset);
  * job id parsed from 'Job <NNN>';
  * status via `bjobs -noheader -o stat -J <name>`, FAIL-CLOSED (a failed bjobs
    returns UNKNOWN and never reports DONE) so we never falsely finalize.

Runs on the submit host (bmiclusterp-head). Generates a small job script per
config that pins thread counts to the reserved slots and runs `run.py`.
"""
from __future__ import annotations
import os
import re
import shlex
import subprocess
from shutil import which

_JOBID = re.compile(r"Job <(\d+)>")


def have_bsub() -> bool:
    return which("bsub") is not None


def require_bsub() -> None:
    if not have_bsub():
        raise SystemExit("ERROR: 'bsub' not found — run on the LSF submit host "
                         "(e.g. bmiclusterp-head), not a compute node or your laptop.")


def _qopt(queue):
    return ["-q", queue] if queue else []


def write_job_script(path, pipeline_dir, python_bin, run_args, threads) -> str:
    """Write a job script that pins BLAS/sklearn threads to the LSF slots, then runs run.py."""
    t = str(int(threads))
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"export OMP_NUM_THREADS={t}",
        f"export OPENBLAS_NUM_THREADS={t}",
        f"export MKL_NUM_THREADS={t}",
        f"export NUMEXPR_NUM_THREADS={t}",
        f"export AMLMM_NJOBS={t}",
        f"cd {shlex.quote(pipeline_dir)}",
        f"{shlex.quote(python_bin)} run.py " + " ".join(shlex.quote(a) for a in run_args),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def submit(name, script_path, log_dir, threads=4, mem_mb=16000, wall="120:00",
           queue=None, depends=None, dry_run=False) -> dict:
    out = os.path.join(log_dir, f"{name}.lsf.out")
    err = os.path.join(log_dir, f"{name}.lsf.err")
    argv = ["bsub", "-L", "/bin/bash", *_qopt(queue),
            "-J", name, "-n", str(int(threads)), "-W", str(wall), "-M", str(int(mem_mb)),
            "-R", f"span[hosts=1] rusage[mem={int(mem_mb)}]",
            "-o", out, "-e", err]
    if depends:
        argv += ["-w", depends]
    argv += ["bash", script_path]
    cmd = " ".join(shlex.quote(a) for a in argv)
    if dry_run:
        return {"name": name, "cmd": cmd, "jobid": None, "dry": True}
    p = subprocess.run(argv, capture_output=True, text=True)
    m = _JOBID.search(p.stdout or "")
    return {"name": name, "cmd": cmd, "jobid": (m.group(1) if m else None),
            "stdout": (p.stdout or "").strip(), "stderr": (p.stderr or "").strip(),
            "rc": p.returncode}


def job_stat_by_id(jobid) -> str:
    """LSF status for a specific job ID — unambiguous. Fail-closed: bjobs error -> UNKNOWN."""
    p = subprocess.run(["bjobs", "-noheader", "-o", "stat", str(jobid)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return "UNKNOWN"
    toks = (p.stdout or "").split()
    return toks[-1] if toks else "GONE"


def job_stat(name) -> str:
    """LSF status by job NAME. Fail-closed: bjobs error -> UNKNOWN (never DONE);
    multiple matches -> AMBIGUOUS (a duplicate name must not mask a failure);
    'GONE' = not in the queue -> check outputs instead. Prefer job_stat_by_id."""
    p = subprocess.run(["bjobs", "-noheader", "-o", "stat", "-J", str(name)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return "UNKNOWN"
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    if len(lines) > 1:
        return "AMBIGUOUS"
    toks = lines[0].split() if lines else []
    return toks[-1] if toks else "GONE"
