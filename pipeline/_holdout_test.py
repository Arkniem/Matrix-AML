#!/usr/bin/env python3
"""Held-out test (cluster, COMPUTE node): train Discovery WITHOUT the 29 user-held-out samples, then
Deploy-predict their mutations + subtype and score vs the (unmasked) truth.

The holdout is auto-active: build_context loads pipeline/holdout_samples.txt into ctx.holdout, and
discovery.labels_for_field masks those samples -> they appear in NO training combo and in NO Deploy
reference set. Each held-out sample is predicted with its OWN donor excluded from references too (so a
Diagnosis held out can't be matched to its Relapse left in training). Truth for scoring is read via the
UNMASKED _labels_for_field_raw. Results -> runs/holdout_test/_results.txt (flushed; LSF buffers stdout).
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from amlmm.context import build_context, Config
from amlmm import discovery as D, deploy as DEP

ctx = build_context(Config(run_id="holdout_test"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

samples = ctx.tables["samples"]
emit("holdout masked: %d samples" % len(ctx.holdout))
# verify the mask actually removed them from training labels
sub = D.labels_for_field(ctx, "subtype")
leak = [s for s in ctx.holdout if s in sub.index and pd.notna(sub.get(s))]
emit("held-out samples with a NON-masked subtype training label (must be 0): %d" % len(leak))

TEST_FIELDS = ["subtype", "mut_NPM1", "mut_FLT3", "mut_TET2", "mut_TP53", "mut_DNMT3A", "mut_NRAS",
               "mut_IDH2", "mut_IDH1", "mut_RUNX1", "mut_SRSF2", "mut_ASXL1", "cyto_inv16"]

emit("\ntraining TRAIN-ONLY Discovery: %d fields x [composition,RNA] x top-4 states, screen10/final20 ..."
     % len(TEST_FIELDS))
cfg = D.DiscoveryConfig(screen_permutations=10, final_permutations=20)
D.run_discovery(ctx, cfg, fields=TEST_FIELDS, modalities=["composition", "RNA"],
                cell_states_top=4, verbose=False)
dr = D.load_discovery(ctx.run_dir)
emit("train-only sweep: %d weight rows | %d assoc rows | significant=%d"
     % (len(dr.weights), len(dr.associations),
        int((dr.weights["weight"] > 0).sum()) if len(dr.weights) else 0))

def _best_thr(scores, y):
    """Decision threshold maximizing balanced accuracy on TRAINING scores (never the test)."""
    s = np.asarray(scores, float)
    best, bt = -1.0, 0.5
    for t in sorted(set(s.tolist())) + [1.01]:
        ba = balanced_accuracy_score(y, (s >= t).astype(int))
        if ba > best:
            best, bt = ba, t
    return bt

def _deploy_scores(subjects, F, truth):
    """Deploy each subject (own donor excluded) -> (present-probability, binary truth) lists."""
    sc, y = [], []
    for s in subjects:
        dn = str(samples["donor_group"].get(s))
        r = DEP.deploy_field(ctx, dr, F, s, dcfg, exclude_donor=dn)
        if r.get("status") == "ok":
            sc.append(r["class_probabilities"].get("present", 0.0))
            y.append(1 if str(truth[s]) == "present" else 0)
    return sc, y

emit("\n=== HELD-OUT PREDICTION (Deploy per-cell-state; refs train-only + own-donor excluded) ===")
emit("%-12s %3s %3s  %5s %6s   %s" % ("field", "n", "pos", "AUC", "ba@.5", "detail"))
dcfg = DEP.DeployConfig(modalities=("composition", "RNA"), k=7, max_combos_per_field=8)
aucs = []
for F in TEST_FIELDS:
    truth = D._labels_for_field_raw(ctx, F)
    held = [s for s in sorted(ctx.holdout) if s in truth.index and pd.notna(truth[s])]
    if not held:
        emit("%-12s  (no held-out truth)" % F); continue
    binary = F.startswith(("mut_", "cyto_"))
    if not binary:                                    # subtype: multiclass accuracy
        yt, yp = [], []
        for s in held:
            dn = str(samples["donor_group"].get(s))
            r = DEP.deploy_field(ctx, dr, F, s, dcfg, exclude_donor=dn)
            if r.get("status") == "ok":
                yt.append(str(truth[s])); yp.append(r["leading"])
        if yt:
            emit("%-12s %3d %3s  %5s %6s %6.3f   acc=%.3f classes=%s"
                 % (F, len(yt), "-", "-", "-", balanced_accuracy_score(yt, yp), accuracy_score(yt, yp), sorted(set(yt))))
        continue
    ysc_te, yt = _deploy_scores(held, F, truth)
    if not yt or len(set(yt)) < 2:
        emit("%-12s %3d %3d  (single-class / no channel)" % (F, len(yt), sum(yt) if yt else 0)); continue
    auc = roc_auc_score(yt, ysc_te)
    ba05 = balanced_accuracy_score(yt, (np.asarray(ysc_te) >= 0.5).astype(int))
    emit("%-12s %3d %3d  %5.2f %6.3f" % (F, len(yt), sum(yt), auc, ba05))
    aucs.append(auc)

if aucs:
    emit("\nmean Deploy (per-cell-state) held-out AUC over %d mutations: %.3f" % (len(aucs), float(np.mean(aucs))))
emit("\nHOLDOUT TEST OK")
