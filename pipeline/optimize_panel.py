#!/usr/bin/env python3
"""Per-mutation OPTIMIZED predictor + sealed held-out evaluation.

For each mutation, pick the best MODEL by train-only donor-grouped CV reliability, then combine
modalities with a CV-gated consistency-weighted late fusion:
  * model choice   = argmax over models of  sum_w(cv_mean)  [w = max(0,cv_mean-.5)/(cv_std+.05)]
  * fusion         = SINGLE dominant modality if CV says one modality dominates (top cv_mean -
                     2nd cv_mean >= GATE), else consistency-weighted rank-blend of all modalities.
Leakage-clean: EVERY selection (model, gate) uses CV only; the sealed held-out samples are scored once.

Reports, per mutation: chosen model, strategy, held-out n, and held-out AUC under
  best-single(CV) | consistency-blend | OPTIMIZED(gated) | oracle(non-deployable, picks on test).
-> runs/single_modality/_optimized.txt + optimized_panel.tsv + heldout_calls.tsv
"""
import os, sys, glob, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "runs", "single_modality")
RES = os.path.join(RUN, "_optimized.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True); open(RES, "a", encoding="utf-8").write(str(m) + "\n")

GATE = 0.08   # a-priori CV-dominance margin (top vs 2nd cv_mean) to deploy a single modality

P = {}
for f in sorted(glob.glob(os.path.join(RUN, "preds_*.pkl"))):
    d = pickle.load(open(f, "rb")); P[d["modality"]] = d
MODS = list(P.keys())
muts = sorted({m for d in P.values() for m in d["data"]})
models = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]
emit("modalities: %s" % ", ".join(MODS))
emit("models: %s" % ", ".join(models))
emit("gate margin (CV dominance -> single modality): %.2f\n" % GATE)


def info(mut, model):
    """[(mod, score_map|None, truth_map, cv_mean|None, cv_std|None)] for mods that ran (mut, model)."""
    av = []
    for mod in MODS:
        rec = P[mod]["data"].get(mut)
        if not rec:
            continue
        mr = rec["models"].get(model)
        if not mr:
            continue
        sc = None if mr.get("scores") is None else dict(zip(rec["test_ids"], mr["scores"]))
        tr = dict(zip(rec["test_ids"], rec["truth"]))
        av.append((mod, sc, tr, mr.get("cv_mean"), mr.get("cv_std")))
    return av


def cv_rank(av):
    """(weighted cv reliability score, [(mod, cv_mean, cv_std) sorted by cv_mean desc])."""
    cms = [(mod, cm, (cs if cs is not None else 0.25)) for mod, sc, tr, cm, cs in av if cm is not None]
    if not cms:
        return None, []
    w = [max(0.0, cm - 0.5) / (cs + 0.05) for _, cm, cs in cms]
    sw = sum(w)
    score = (sum(wi * cm for wi, (_, cm, _) in zip(w, cms)) / sw) if sw > 0 else float(np.mean([c for _, c, _ in cms]))
    return score, sorted(cms, key=lambda t: -t[1])


def heldout(av, strategy, cm_sorted):
    """Held-out AUC of (model's av) under 'single' (CV-top modality) or 'blend' (consistency-weighted).
    Per-sample percentile fusion over the modalities that scored each sample -> keeps all held-out n."""
    rows = {}                                              # sample -> (truth, {mod: pct, cv})
    cmmap = {m: cm for m, cm, cs in cm_sorted}
    csmap = {m: cs for m, cm, cs in cm_sorted}
    for mod, sc, tr, cm, cs in av:
        if sc is None or cm is None:
            continue
        ids = [s for s in sc]; vals = np.array([sc[s] for s in ids], float)
        if len(ids) < 2:
            continue
        pct = (rankdata(vals) - 1) / (len(ids) - 1)        # 0..1 percentile within this modality's held-out set
        for s, p in zip(ids, pct):
            rows.setdefault(s, [tr[s], {}])[1][mod] = p
    if strategy == "single":
        top = cm_sorted[0][0]
        use = {top: 1.0}
    else:
        use = {m: max(0.0, cmmap[m] - 0.5) / (csmap[m] + 0.05) for m in cmmap}
        if sum(use.values()) <= 0:
            use = {m: 1.0 for m in cmmap}
    y, s_, ids = [], [], []
    for samp, (truth, pm) in rows.items():
        ws = {m: use[m] for m in pm if use.get(m, 0) > 0}
        if not ws:
            continue
        y.append(int(truth)); ids.append(samp)
        s_.append(sum(ws[m] * pm[m] for m in ws) / sum(ws.values()))
    if len(set(y)) < 2:
        return np.nan, len(y), ids, y, s_
    return roc_auc_score(y, s_), len(y), ids, y, s_


prows, call_rows = [], []
for mut in muts:
    # CV-only model selection
    best = (-1, None, [], None)
    cache = {}
    for model in models:
        av = info(mut, model); cache[model] = av
        sc, cms = cv_rank(av)
        if sc is None:
            continue
        if sc > best[0]:
            best = (sc, model, cms, av)
    cvscore, model_star, cms, av = best
    if model_star is None:
        continue
    dominant = len(cms) >= 2 and (cms[0][1] - cms[1][1] >= GATE)
    strat = "single" if dominant else "blend"

    auc_single, n, *_ = heldout(av, "single", cms)
    auc_blend, _, _, _, _ = heldout(av, "blend", cms)
    auc_opt, n_opt, ids, y, sopt = heldout(av, strat, cms)

    # oracle (NON-deployable): best (model, strategy) on the held-out set itself
    oracle = -1; orc = ("", "")
    for model in models:
        for st in ("single", "blend"):
            a, _, _, _, _ = heldout(cache[model], st, cv_rank(cache[model])[1])
            if a == a and a > oracle:
                oracle = a; orc = (model, st)

    prows.append((mut, model_star, strat, cms[0][0], n_opt, auc_single, auc_blend, auc_opt, oracle))
    for s, t, sc in zip(ids, y, sopt):
        call_rows.append((mut, s, int(t), round(float(sc), 3)))

df = pd.DataFrame(prows, columns=["mutation", "model", "strategy", "lead_modality", "n",
                                  "single", "blend", "optimized", "oracle"])
df.to_csv(os.path.join(RUN, "optimized_panel.tsv"), sep="\t", index=False)
pd.DataFrame(call_rows, columns=["mutation", "sample", "truth", "opt_score"]).to_csv(
    os.path.join(RUN, "heldout_calls.tsv"), sep="\t", index=False)

emit("%-14s %-8s %-7s %-11s %3s %7s %7s %9s %8s" % (
    "mutation", "model", "strat", "lead_mod", "n", "single", "blend", "OPTIMIZED", "oracle*"))
for r in prows:
    emit("%-14s %-8s %-7s %-11s %3d %7.2f %7.2f %9.2f %8.2f" % (
        r[0].replace("mut_", "").replace("cyto_", "")[:14], r[1], r[2], r[3][:11],
        r[4], r[5], r[6], r[7], r[8]))
emit("%-14s %-8s %-7s %-11s %3s %7.3f %7.3f %9.3f %8.3f" % (
    "MEAN", "", "", "", "", df.single.mean(), df.blend.mean(), df.optimized.mean(), df.oracle.mean()))
emit("\noptimized vs best-single(CV): %+.3f | vs consistency-blend: %+.3f | headroom to oracle: %+.3f"
     % (df.optimized.mean() - df.single.mean(), df.optimized.mean() - df.blend.mean(),
        df.oracle.mean() - df.optimized.mean()))

# per-held-out-sample summary: rank-based call correctness across its labeled mutations
emit("\nper-held-out-sample (rank-call vs truth across mutations):")
cd = pd.DataFrame(call_rows, columns=["mutation", "sample", "truth", "opt_score"])
ssum = []
for samp, g in cd.groupby("sample"):
    ssum.append((samp, len(g), int(g.truth.sum())))
emit("  %d held-out samples scored across %d mutations (%d total sample-mutation tests)"
     % (cd["sample"].nunique(), cd["mutation"].nunique(), len(cd)))
emit("\nOPTIMIZE OK")
