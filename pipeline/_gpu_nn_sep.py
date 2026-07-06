#!/usr/bin/env python3
"""Separate per-mutation models (no multi-task sharing) vs the multi-task NN. For each mutation, train
(a) a single-task NN and (b) an L2-logistic baseline on composition + RNA marker genes, holdout-masked
(rich explicit labels), and score held-out AUC. Tests whether dropping the shared representation +
simpler models beat the multi-task NN (0.548). Run on gpu-b6000. Results -> runs/gpu_nn_sep/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio

ctx = build_context(Config(run_id="gpu_nn_sep"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emit("device=%s | torch=%s | holdout=%d" % (dev, torch.__version__, len(ctx.holdout)))
    samples = ctx.tables["samples"]; sk_all = list(samples.index)

    comp = D._sample_level_matrix(ctx, "composition", set(sk_all))
    markers = pio.udon_markers(ctx, "RNA")
    rna = dataio.sample_modality_matrix(ctx, "RNA")
    rna = np.log1p(rna[[g for g in markers if g in rna.columns]].clip(lower=0))
    feat = comp.join(rna, how="inner").dropna()
    emit("features: %d (composition %d + RNA-markers %d) | samples %d"
         % (feat.shape[1], comp.shape[1], rna.shape[1], feat.shape[0]))

    MUTS = ["mut_NPM1", "mut_FLT3", "mut_TET2", "mut_TP53", "mut_DNMT3A", "mut_NRAS",
            "mut_IDH2", "mut_IDH1", "mut_RUNX1", "mut_SRSF2", "cyto_inv16"]
    MT = {"mut_NPM1": .65, "mut_FLT3": .56, "mut_TET2": .63, "mut_TP53": .58, "mut_DNMT3A": .45,
          "mut_NRAS": .50, "mut_IDH2": .46, "mut_IDH1": .55, "mut_RUNX1": .51, "mut_SRSF2": .32,
          "cyto_inv16": .95}

    def to01(s):
        return s.map({"present": 1.0, "absent": 0.0})

    class Net(nn.Module):
        def __init__(self, nin):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nin, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x).squeeze(1)

    def train_nn(Xtr, ytr, donor, Xte):
        rng = np.random.RandomState(0); ud = rng.permutation(np.unique(donor))
        val_d = set(ud[:max(1, len(ud) // 5)]); vm = np.array([d in val_d for d in donor])
        torch.manual_seed(0)
        m = Net(Xtr.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        bce = nn.BCEWithLogitsLoss()
        Xg = torch.tensor(Xtr).to(dev); yg = torch.tensor(ytr).to(dev)
        tr = torch.tensor(~vm).to(dev); va = torch.tensor(vm).to(dev)
        best, bs, wait = 1e9, None, 0
        for ep in range(500):
            m.train(); opt.zero_grad(); loss = bce(m(Xg[tr]), yg[tr]); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = bce(m(Xg[va]), yg[va]).item()
            if vl < best - 1e-4:
                best, bs, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                wait += 1
            if wait > 40:
                break
        m.load_state_dict(bs); m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(Xte).to(dev))).cpu().numpy()

    emit("\n%-12s %4s %4s  %6s %6s %6s   (held-out AUC)" % ("mutation", "n", "pos", "sepNN", "logreg", "multiNN"))
    sep_nn, sep_lr = [], []
    for m in MUTS:
        yall = to01(D._labels_for_field_raw(ctx, m)).reindex(feat.index)
        ymask = to01(D.labels_for_field(ctx, m)).reindex(feat.index)   # masked (holdout NaN)
        train = [s for s in feat.index if pd.notna(ymask[s]) and s not in ctx.holdout]
        test = [s for s in feat.index if s in ctx.holdout and pd.notna(yall[s])]
        if len(test) < 4 or pd.Series([yall[s] for s in train]).nunique() < 2:
            emit("%-12s  (insufficient)" % m); continue
        sc = StandardScaler().fit(feat.loc[train].values)
        Xtr = sc.transform(feat.loc[train].values).astype(np.float32)
        Xte = sc.transform(feat.loc[test].values).astype(np.float32)
        ytr = np.array([yall[s] for s in train], dtype=np.float32)
        yte = np.array([int(yall[s]) for s in test])
        if len(set(yte)) < 2:
            emit("%-12s %4d %4d  (single-class held-out)" % (m, len(test), int(yte.sum()))); continue
        donor = samples.reindex(train)["donor_group"].astype(str).values
        p_nn = train_nn(Xtr, ytr, donor, Xte)
        lr = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
        lr.fit(Xtr, ytr.astype(int)); p_lr = lr.predict_proba(Xte)[:, 1]
        a_nn = roc_auc_score(yte, p_nn); a_lr = roc_auc_score(yte, p_lr)
        emit("%-12s %4d %4d  %6.2f %6.2f %6.2f" % (m, len(test), int(yte.sum()), a_nn, a_lr, MT.get(m, float("nan"))))
        sep_nn.append(a_nn); sep_lr.append(a_lr)
    if sep_nn:
        emit("\nmean held-out AUC | separate NN %.3f | logreg %.3f | multi-task NN %.3f"
             % (float(np.mean(sep_nn)), float(np.mean(sep_lr)),
                float(np.mean([MT[m] for m in MUTS if m in MT]))))
    emit("\nSEP NN OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
