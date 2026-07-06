#!/usr/bin/env python3
"""Deploy v1 validation against the disc_sweep Discovery corpus (run on an LSF COMPUTE node).

A) composition-only LEAVE-ONE-DONOR-OUT subtype accuracy — the decisive mechanism check: does
   weight-moderated Spearman NN-transfer recover subtype on held-out donors, clearly above the
   permutation chance ceiling (~0.29 for 5 grouped classes)? Compared to Discovery's composition OOF.
B) full certified aggregate (composition + RNA) on subtype exemplars — leading call + per-channel detail.
C) UDON Spearman signature on exemplars — neighbors' subtypes per cell-state + consensus.

Results are written to runs/deploy_selftest/_results.txt with flush (LSF buffers a job's stdout until
exit, so the direct file is what the poller watches). Parts B/C are wrapped so an RNA hiccup never loses
the decisive Part A.
"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from amlmm.context import build_context, Config
from amlmm import deploy as DEP, discovery as D

warnings.filterwarnings("ignore")
SWEEP = os.environ.get("DISC_SWEEP",
                       "/data/salomonis-archive/LabFiles/Nicholas/AML-multimodal/runs/disc_sweep")
ctx = build_context(Config(run_id="deploy_selftest"))
RES = ctx.path("_results.txt")
open(RES, "w").close()


def emit(msg=""):
    print(msg, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


samples = ctx.tables["samples"]
dr = D.load_discovery(SWEEP)
emit("loaded discovery: %d weight rows | %d assoc rows | %s" % (len(dr.weights), len(dr.associations), dr.run_dir))

A = dr.associations_for("subtype", "composition", "__sample__")
assert not A.empty, "no subtype/composition associations in the sweep"
truth_of = {str(r["pseudobulk_id"]): str(r["true"]) for _, r in A.iterrows()}

# ---------- A) composition-only leave-one-donor-out ----------
emit("\n==== A) composition-only leave-one-donor-out: subtype ====")
cfgA = DEP.DeployConfig(modalities=("composition",), k=5)
yt, yp = [], []
for sk, truth in truth_of.items():
    dn = str(samples["donor_group"].get(sk))
    r = DEP.deploy_field(ctx, dr, "subtype", sk, cfgA, exclude_donor=dn)
    if r.get("status") == "ok":
        yt.append(truth); yp.append(r["leading"])
ba = balanced_accuracy_score(yt, yp); acc = accuracy_score(yt, yp)
fa = dr.field_ability().get("subtype", {})
emit("  NN-transfer LODO: balanced_acc=%.3f  acc=%.3f  (n=%d, classes=%s)" % (ba, acc, len(yt), sorted(set(yt))))
emit("  Discovery best (RF OOF): ba=%s via %s@%s" % (fa.get("best_balanced_accuracy"),
     fa.get("best_modality"), fa.get("best_cell_state")))
emit("  5-class grouped chance ceiling p95~0.29 -> NN is %s" %
     ("ABOVE chance" if ba > 0.40 else "NOT clearly above chance (!)"))

exemplars = A.groupby("true").head(2)["pseudobulk_id"].astype(str).tolist()[:8]

# ---------- B) full certified aggregate (composition + RNA) ----------
emit("\n==== B) certified aggregate (composition + RNA) on exemplars ====")
try:
    cfgB = DEP.DeployConfig(modalities=("composition", "RNA"), k=5, max_combos_per_field=8, use_udon=False)
    for sk in exemplars:
        dn = str(samples["donor_group"].get(sk))
        cc = DEP.deploy_field(ctx, dr, "subtype", sk, cfgB, exclude_donor=dn)
        if cc.get("status") == "ok":
            chans = ", ".join("%s@%s:%s(w%.2f)" % (c["modality"], c["cell_state"][:10], c["pred"], c["weight"])
                              for c in cc["channels"][:4])
            emit("  %-34s true=%-7s -> %-7s conf=%.2f margin=%.2f | %dch: %s"
                 % (sk[:34], truth_of[sk], cc["leading"], cc["confidence"], cc["margin"], cc["n_channels"], chans))
        else:
            emit("  %-34s true=%-7s -> %s" % (sk[:34], truth_of[sk], cc.get("status")))
except Exception as e:
    emit("  [B FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())

# ---------- C) UDON Spearman signature ----------
emit("\n==== C) UDON signature (descriptive) on exemplars ====")
try:
    cfgC = DEP.DeployConfig(use_udon=True, udon_k=5, udon_cell_states_top=4)
    for sk in exemplars[:5]:
        u = DEP.udon_subtype_signature(ctx, sk, cfgC)
        if u.get("status") == "ok":
            ps = "; ".join("%s:%s(%.2f)" % (s["cell_state"][:10], s["call"], s["prob"]) for s in u["per_state"][:4])
            emit("  %-34s true=%-7s -> UDON consensus=%s(%.2f) | %s"
                 % (sk[:34], truth_of[sk], u["consensus_call"], u["consensus_prob"], ps))
        else:
            emit("  %-34s true=%-7s -> UDON %s: %s" % (sk[:34], truth_of[sk], u.get("status"), u.get("reason")))
except Exception as e:
    emit("  [C FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())

# ---------- artifact ----------
try:
    sk0 = exemplars[0]; dn0 = str(samples["donor_group"].get(sk0))
    rep = DEP.run_deploy_atlas(ctx, dr, sk0, ["subtype"], DEP.DeployConfig(use_udon=True), exclude_donor=dn0)
    ctx.save_json(rep, "deploy_report_example.json")
    cc0 = rep["certified_calls"]["subtype"]
    emit("\nexample report: %s true=%s -> certified %s (%.2f), udon %s"
         % (sk0, truth_of[sk0], cc0.get("leading"), cc0.get("confidence", 0.0),
            rep["udon_signature"].get("consensus_call")))
    emit("wrote runs/deploy_selftest/deploy_report_example.json")
except Exception as e:
    emit("  [artifact FAILED] %s: %s" % (type(e).__name__, e))

emit("\nDEPLOY SELFTEST OK")
