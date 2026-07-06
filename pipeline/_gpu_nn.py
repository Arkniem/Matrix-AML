#!/usr/bin/env python3
"""GPU multi-task NN (the experiment): predict mutations from composition + RNA marker genes, trained
WITHOUT the 29 held-out samples (holdout mask), with MASKED BCE multi-task heads — each sample trains
only the mutation heads it actually has a label for (the fair multi-label handling sklearn couldn't do).
Heavy regularization (dropout + weight decay) + donor-grouped early stopping, because n~370 << features.
Scores held-out AUC vs Deploy's composition channel. Run on gpu-b6000. Results -> runs/gpu_nn/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio

ctx = build_context(Config(run_id="gpu_nn"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emit("device=%s | torch=%s | gpu=%s | holdout=%d"
         % (dev, torch.__version__, (torch.cuda.get_device_name(0) if dev == "cuda" else "-"), len(ctx.holdout)))
    samples = ctx.tables["samples"]; sk_all = list(samples.index)

    # ---- features: composition (90) + RNA marker genes (sample-level aggregate, log1p) ----
    comp = D._sample_level_matrix(ctx, "composition", set(sk_all))
    markers = pio.udon_markers(ctx, "RNA")
    rna = dataio.sample_modality_matrix(ctx, "RNA")
    rna = rna[[g for g in markers if g in rna.columns]]
    rna = np.log1p(rna.clip(lower=0))
    feat = comp.join(rna, how="inner").dropna()
    emit("features: composition %d + RNA-markers %d = %d | samples %d"
         % (comp.shape[1], rna.shape[1], feat.shape[1], feat.shape[0]))

    MUTS = ["mut_NPM1", "mut_FLT3", "mut_TET2", "mut_TP53", "mut_DNMT3A", "mut_NRAS",
            "mut_IDH2", "mut_IDH1", "mut_RUNX1", "mut_SRSF2", "mut_ASXL1", "cyto_inv16"]

    def to01(s):
        return s.map({"present": 1.0, "absent": 0.0})
    Ytr = pd.DataFrame({m: to01(D.labels_for_field(ctx, m)) for m in MUTS}).reindex(feat.index)   # masked
    Yall = pd.DataFrame({m: to01(D._labels_for_field_raw(ctx, m)) for m in MUTS}).reindex(feat.index)  # truth
    Mtr = Ytr.notna().astype(float)

    train = [s for s in feat.index if Mtr.loc[s].sum() > 0 and s not in ctx.holdout]
    test = [s for s in feat.index if s in ctx.holdout]
    emit("train %d | test(held-out) %d | mutation heads %d" % (len(train), len(test), len(MUTS)))

    scaler = StandardScaler().fit(feat.loc[train].values)
    Xtr = scaler.transform(feat.loc[train].values).astype(np.float32)
    Xte = scaler.transform(feat.loc[test].values).astype(np.float32)
    Ytr_v = Ytr.loc[train].fillna(0).values.astype(np.float32)
    Mtr_v = Mtr.loc[train].values.astype(np.float32)

    # donor-grouped val split (early stopping)
    donor = samples.reindex(train)["donor_group"].astype(str).values
    rng = np.random.RandomState(0); ud = rng.permutation(np.unique(donor))
    val_d = set(ud[:max(1, len(ud) // 5)]); vmask = np.array([d in val_d for d in donor])

    class Net(nn.Module):
        def __init__(self, nin, nout):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nin, 128), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, nout))
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(0)
    model = Net(Xtr.shape[1], len(MUTS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    Xg = torch.tensor(Xtr).to(dev); Yg = torch.tensor(Ytr_v).to(dev); Mg = torch.tensor(Mtr_v).to(dev)
    tr_s = torch.tensor(~vmask).to(dev); va_s = torch.tensor(vmask).to(dev)

    def mloss(sel):
        out = model(Xg[sel]); l = bce(out, Yg[sel]) * Mg[sel]
        return l.sum() / Mg[sel].sum().clamp(min=1)

    best, best_state, wait = 1e9, None, 0
    for ep in range(600):
        model.train(); opt.zero_grad(); loss = mloss(tr_s); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = mloss(va_s).item()
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait > 50:
            break
    model.load_state_dict(best_state); model.eval()
    emit("trained: stopped at epoch %d, val masked-BCE %.4f" % (ep, best))
    with torch.no_grad():
        P = torch.sigmoid(model(torch.tensor(Xte).to(dev))).cpu().numpy()

    emit("\n%-12s %4s %4s %8s   (held-out, explicit labels)" % ("mutation", "n", "pos", "NN-AUC"))
    nn_a = []
    for j, m in enumerate(MUTS):
        yte = Yall.loc[test, m].values
        ok = ~pd.isna(yte)
        yv = yte[ok].astype(int); pv = P[ok, j]
        if len(set(yv)) < 2:
            emit("%-12s %4d %4d   (single-class)" % (m, int(ok.sum()), int(yv.sum()) if len(yv) else 0)); continue
        a = roc_auc_score(yv, pv)
        emit("%-12s %4d %4d %8.2f" % (m, len(yv), int(yv.sum()), a)); nn_a.append(a)
    if nn_a:
        emit("\nmulti-task NN mean held-out AUC over %d mutations: %.3f" % (len(nn_a), float(np.mean(nn_a))))
    emit("\nGPU NN OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
