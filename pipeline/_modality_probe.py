#!/usr/bin/env python3
"""Characterize every modality before the multi-modality experiment: dimensionality,
sample coverage, held-out (sealed-29) coverage, and imputation fidelity (heldout_spearman)
distribution. Tells us which modalities are 'feature-limited' (no FS needed) per the boss.
-> runs/modality_probe/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from amlmm.context import build_context, Config
from amlmm import dataio, discovery as D

ctx = build_context(Config(run_id="modality_probe"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    samples = ctx.tables["samples"]
    hold = sorted(ctx.holdout)
    emit("samples total: %d | sealed held-out: %d" % (len(samples), len(hold)))
    emit("layout: %s\n" % ctx.layout)

    emit("%-14s %8s %8s %9s %9s   %s" % ("modality", "nfeat", "nsamp", "held/29", "fid>=.3", "fidelity p10/50/90"))
    comp = D._sample_level_matrix(ctx, "composition", set(samples.index))
    hc = sum(1 for s in hold if s in comp.index)
    emit("%-14s %8d %8d %9s %9s" % ("composition", comp.shape[1], comp.shape[0], "%d" % hc, "-"))

    for mod in ["RNA", "GRN", "Metabolite", "Lipid", "ADT"]:
        try:
            M = dataio.sample_modality_matrix(ctx, mod)
            held = sum(1 for s in hold if s in M.index)
            fid = dataio.feature_fidelity(ctx, mod)
            if fid is not None:
                fv = fid.dropna().values
                nfid = int((fv >= 0.3).sum())
                p = np.percentile(fv, [10, 50, 90])
                fstr = "%.2f / %.2f / %.2f" % (p[0], p[1], p[2])
            else:
                nfid, fstr = M.shape[1], "(no fidelity var)"
            emit("%-14s %8d %8d %9d %9d   %s" % (mod, M.shape[1], M.shape[0], held, nfid, fstr))
        except Exception as e:
            emit("%-14s  FAILED: %s" % (mod, e))

    # cell-communication is sample-level, separate loader
    try:
        cc = dataio.cellcomm_matrix(ctx)
        held = sum(1 for s in hold if s in cc.index)
        emit("%-14s %8d %8d %9d %9s" % ("cell-comm", cc.shape[1], cc.shape[0], held, "-"))
    except Exception as e:
        emit("cell-comm FAILED: %s" % e)

    # how many held-out are covered by the UNION of non-RNA modalities (composition+ADT+Lipid+Metab+GRN)?
    nonrna = set(comp.index)
    for mod in ["GRN", "Metabolite", "Lipid", "ADT"]:
        try:
            nonrna |= set(dataio.sample_modality_matrix(ctx, mod).index)
        except Exception:
            pass
    emit("\nheld-out covered by composition alone: %d/29" % sum(1 for s in hold if s in comp.index))
    emit("held-out covered by ANY non-RNA modality: %d/29" % sum(1 for s in hold if s in nonrna))
    rnaidx = set(dataio.sample_modality_matrix(ctx, "RNA").index)
    emit("held-out covered by RNA: %d/29" % sum(1 for s in hold if s in rnaidx))
    emit("held-out covered by composition AND all 4 imputed:")
    allmods = {m: set(dataio.sample_modality_matrix(ctx, m).index) for m in ["GRN", "Metabolite", "Lipid", "ADT"]}
    inter = set(comp.index)
    for s in allmods.values():
        inter &= s
    emit("  %d/29 in composition+GRN+Metab+Lipid+ADT all" % sum(1 for s in hold if s in inter))
    for m, idx in allmods.items():
        emit("  %-12s held-out coverage: %d/29" % (m, sum(1 for s in hold if s in idx)))

    emit("\nPROBE OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
