#!/usr/bin/env python3
"""LOCAL holdout test: with the user's 29 highlighted samples held out of training, can the COMPOSITION
channel (the one we can compute locally) predict their mutations via Deploy's Spearman NN-transfer?

This is the measured-channel preview of the held-out test. The strong mutation predictors are RNA
channels (inv16/TET2/NPM1 etc. — see the disc_sweep mutation matrix), which need the cluster; this proves
the holdout framework + shows composition's contribution honestly. Reads inputs from /tmp/deploy_local.
Also writes holdout_samples.txt (the 29 sample_keys) for the pipeline's training mask.
"""
import os, openpyxl
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score

HERE = os.getcwd()   # run from the inputs dir (cd /tmp/deploy_local && python <this>)
XLSX = r"C:\Users\krog5w\Downloads\AML_metadata_CLEAN-selected.xlsx"
MUTS = ["NPM1", "FLT3", "TET2", "TP53", "IDH2", "IDH1", "DNMT3A", "NRAS", "RUNX1",
        "SRSF2", "U2AF1", "inv(16)_CBFB-MYH11"]

# ---- composition keyed to sample_key (replicates dataio) ----
comp = pd.read_csv(os.path.join(HERE, "sample_cellstate_counts.tsv"), sep="\t")
sid = comp.iloc[:, 0].astype(str)
comp = comp.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
lsc = pd.read_csv(os.path.join(HERE, "prediction_subtype_classifications_RF.tsv"), sep="\t", dtype=str)
xwalk = dict(zip(lsc["SampleID"].astype(str), lsc["Dataset"].astype(str) + "::" + lsc["Sample"].astype(str)))
comp.index = [xwalk.get(i, i) for i in sid]
comp = comp[~comp.index.duplicated(keep="first")]
compn = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)   # L1, Spearman-invariant anyway

# ---- mutation labels + held-out set from the highlighted xlsx ----
wb = openpyxl.load_workbook(XLSX)
ws = wb["Clinical_Metadata"]
H = [c.value for c in ws[1]]
idx = {h: i for i, h in enumerate(H)}
held, recs = set(), []
for r in ws.iter_rows(min_row=2):
    ds, sm = r[idx["Dataset"]].value, r[idx["Sample"]].value
    if ds is None:
        continue
    sk = "%s::%s" % (ds, sm)
    if getattr(r[idx["Sample"]].fill, "patternType", None):
        held.add(sk)
    rec = {"sk": sk}
    for m in MUTS:
        v = r[idx[m]].value if m in idx else None
        rec[m] = 1 if str(v) in ("1", "1.0") else (0 if str(v) in ("0", "0.0") else np.nan)
    recs.append(rec)
lab = pd.DataFrame(recs).drop_duplicates("sk").set_index("sk")
with open(os.path.join(HERE, "holdout_samples.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(held)) + "\n")
print("held-out: %d samples | with composition: %d" % (len(held), sum(s in compn.index for s in held)))

# ---- Deploy NN primitives (verbatim from deploy.py) ----
def s2e(q, R):
    q = rankdata(np.asarray(q, float)); R = rankdata(np.asarray(R, float), axis=1)
    qc = q - q.mean(); qn = qc / (np.linalg.norm(qc) or 1.0)
    Rc = R - R.mean(axis=1, keepdims=True); Rn = Rc / (np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12)
    return Rn @ qn

def nn_prob_pos(q, refX, reflab, k=7):
    corr = s2e(q, refX); order = np.argsort(-corr)[:k]
    num, den = 0.0, 0.0
    for j in order:
        w = max(float(corr[j]), 0.0); num += w * float(reflab[j]); den += w
    return (num / den) if den > 0 else float(np.mean([reflab[j] for j in order]))

# ---- per-mutation held-out prediction (train = non-held, test = held) ----
print("\n%-22s %5s %4s %4s  %8s %8s %8s" % ("mutation", "nTest", "pos", "base", "bal_acc", "AUC", "pos_recall"))
from sklearn.metrics import roc_auc_score
rows = []
for m in MUTS:
    y = lab[m].dropna()
    usable = [s for s in y.index if s in compn.index]
    tr = [s for s in usable if s not in held]
    te = [s for s in usable if s in held]
    if len(te) < 3 or pd.Series([y[s] for s in tr]).nunique() < 2:
        continue
    refX = compn.loc[tr].values; reflab = np.array([y[s] for s in tr])
    probs = np.array([nn_prob_pos(compn.loc[s].values, refX, reflab) for s in te])
    truth = np.array([int(y[s]) for s in te])
    pred = (probs >= 0.5).astype(int)
    base = truth.mean()
    try:
        auc = roc_auc_score(truth, probs) if truth.min() != truth.max() else float("nan")
    except Exception:
        auc = float("nan")
    ba = balanced_accuracy_score(truth, pred) if truth.min() != truth.max() else float("nan")
    posrec = (pred[truth == 1].mean() if (truth == 1).any() else float("nan"))
    print("%-22s %5d %4d %.2f  %8.3f %8s %8s"
          % (m, len(te), int(truth.sum()), base, ba,
             ("%.3f" % auc) if auc == auc else "  n/a",
             ("%.2f" % posrec) if posrec == posrec else " n/a"))
    rows.append((m, auc, ba))
ok = [r for r in rows if r[1] == r[1]]
if ok:
    print("\nmean AUC over %d testable mutations (composition channel): %.3f"
          % (len(ok), np.mean([r[1] for r in ok])))
print("\nNOTE: composition is a WEAK mutation predictor (disc_sweep: composition mut-weights ~0.3-0.4);")
print("the STRONG channels are RNA (inv16 1.0, TET2 0.98, etc.) — those run on the cluster (full test).")
print("HOLDOUT LOCAL TEST OK")
