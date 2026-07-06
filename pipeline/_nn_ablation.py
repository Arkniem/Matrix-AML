#!/usr/bin/env python3
"""NN ablation: include the neural nets in the comparison, on {raw, batch-corrected} features.
Multi-task NN (shared body + masked-BCE per-mutation heads) and separate per-mutation NN, holdout-masked
rich labels, sealed 29. CPU torch (no GPU slot needed). -> runs/nn_ablation/_results.txt
"""
import os, sys, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from amlmm.context import build_context, Config
from amlmm import discovery as D, dataio, pseudobulk_io as pio, genetics

ctx = build_context(Config(run_id="nn_ablation"))
RES = ctx.path("_results.txt"); open(RES, "w").close()
def emit(m=""):
    print(m, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

try:
    import torch, torch.nn as nn
    torch.set_num_threads(2)
    samples = ctx.tables["samples"]
    comp = D._sample_level_matrix(ctx, "composition", set(samples.index))
    markers = pio.udon_markers(ctx, "RNA")
    rna = dataio.sample_modality_matrix(ctx, "RNA")
    rna = np.log1p(rna[[g for g in markers if g in rna.columns]].clip(lower=0))
    feat_raw = comp.join(rna, how="inner").dropna()
    ds = samples["dataset"].astype(str); hold = set(ctx.holdout)
    pool = [s for s in feat_raw.index if s not in hold]; gmean = feat_raw.loc[pool].mean(axis=0)
    feat_bc = feat_raw.copy()
    for d in ds.loc[feat_raw.index].unique():
        trd = [s for s in pool if ds.get(s) == d]
        if len(trd) >= 4:
            dm = feat_raw.loc[trd].mean(axis=0); rows = [s for s in feat_raw.index if ds.get(s) == d]
            feat_bc.loc[rows] = feat_raw.loc[rows].values - dm.values + gmean.values
    emit("features %d | raw + batch-corrected" % feat_raw.shape[1])

    M = ctx.tables.get("mutations") or genetics.build_mutation_matrix(ctx)
    _m01 = {"present": 1.0, "absent": 0.0}
    MUTS = []
    for f in sorted(c for c in M.columns if str(c).startswith(("mut_", "cyto_"))):
        y = D._labels_for_field_raw(ctx, f).map(_m01).reindex(feat_raw.index); inh = y.index.isin(hold)
        if int(((inh) & (y == 1)).sum()) >= 3 and int(((inh) & (y == 0)).sum()) >= 3 \
           and int(((~inh) & (y == 1)).sum()) >= 5 and int(((~inh) & (y == 0)).sum()) >= 5:
            MUTS.append(f)
    emit("mutations: %d" % len(MUTS))

    Yall = pd.DataFrame({m: D._labels_for_field_raw(ctx, m).map(_m01) for m in MUTS}).reindex(feat_raw.index)
    Ymask = pd.DataFrame({m: D.labels_for_field(ctx, m).map(_m01) for m in MUTS}).reindex(feat_raw.index)
    Mtr = Ymask.notna().astype(float)
    train = [s for s in feat_raw.index if Mtr.loc[s].sum() > 0 and s not in hold]
    test = [s for s in feat_raw.index if s in hold]
    donor = samples.reindex(train)["donor_group"].astype(str).values
    ud = np.random.RandomState(0).permutation(np.unique(donor)); vd = set(ud[:max(1, len(ud) // 5)])
    vm = np.array([d in vd for d in donor])

    class Net(nn.Module):
        def __init__(s, nin, nout):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(nin, 128), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, nout))
        def forward(s, x):
            return s.net(x)

    def multitask(feat):
        sc = StandardScaler().fit(feat.loc[train].values)
        Xtr = torch.tensor(sc.transform(feat.loc[train].values).astype(np.float32))
        Xte = torch.tensor(sc.transform(feat.loc[test].values).astype(np.float32))
        Ytr = torch.tensor(Yall.loc[train, MUTS].fillna(0).values.astype(np.float32))
        Mt = torch.tensor(Mtr.loc[train, MUTS].values.astype(np.float32))
        trs, vas = torch.tensor(~vm), torch.tensor(vm)
        torch.manual_seed(0); model = Net(Xtr.shape[1], len(MUTS))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2); bce = nn.BCEWithLogitsLoss(reduction="none")
        def ml(sel):
            l = bce(model(Xtr[sel]), Ytr[sel]) * Mt[sel]; return l.sum() / Mt[sel].sum().clamp(min=1)
        best, bs, w = 1e9, None, 0
        for ep in range(500):
            model.train(); opt.zero_grad(); ml(trs).backward(); opt.step(); model.eval()
            with torch.no_grad():
                vl = ml(vas).item()
            if vl < best - 1e-4:
                best, bs, w = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                w += 1
            if w > 40:
                break
        model.load_state_dict(bs); model.eval()
        with torch.no_grad():
            P = torch.sigmoid(model(Xte)).numpy()
        out = {}
        for j, m in enumerate(MUTS):
            yt = Yall.loc[test, m].values; ok = ~pd.isna(yt); yv = yt[ok].astype(int)
            if len(set(yv)) > 1:
                out[m] = roc_auc_score(yv, P[ok, j])
        return out

    def separate(feat):
        out = {}
        for m in MUTS:
            y, ym = Yall[m], Ymask[m]
            tr = [s for s in feat.index if pd.notna(ym[s]) and s not in hold]
            te = [s for s in feat.index if s in hold and pd.notna(y[s])]
            yte = np.array([int(y[s]) for s in te])
            if len(set(yte)) < 2:
                continue
            sc = StandardScaler().fit(feat.loc[tr].values)
            Xtr = torch.tensor(sc.transform(feat.loc[tr].values).astype(np.float32))
            Xte = torch.tensor(sc.transform(feat.loc[te].values).astype(np.float32))
            ytr = torch.tensor(np.array([y[s] for s in tr], dtype=np.float32))
            dn = samples.reindex(tr)["donor_group"].astype(str).values
            u2 = np.random.RandomState(0).permutation(np.unique(dn)); v2 = set(u2[:max(1, len(u2) // 5)])
            vmm = np.array([d in v2 for d in dn]); trs, vas = torch.tensor(~vmm), torch.tensor(vmm)
            torch.manual_seed(0); md = Net(Xtr.shape[1], 1)
            op = torch.optim.AdamW(md.parameters(), lr=1e-3, weight_decay=1e-2); bce = nn.BCEWithLogitsLoss()
            best, bs, w = 1e9, None, 0
            for ep in range(400):
                md.train(); op.zero_grad(); bce(md(Xtr[trs]).squeeze(1), ytr[trs]).backward(); op.step(); md.eval()
                with torch.no_grad():
                    vl = bce(md(Xtr[vas]).squeeze(1), ytr[vas]).item()
                if vl < best - 1e-4:
                    best, bs, w = vl, {k: v.clone() for k, v in md.state_dict().items()}, 0
                else:
                    w += 1
                if w > 30:
                    break
            md.load_state_dict(bs); md.eval()
            with torch.no_grad():
                out[m] = roc_auc_score(yte, torch.sigmoid(md(Xte).squeeze(1)).numpy())
        return out

    C = {}
    for fname, feat in [("raw", feat_raw), ("bc", feat_bc)]:
        C[("MT", fname)] = multitask(feat); emit("  multitask/%s done" % fname)
        C[("sep", fname)] = separate(feat); emit("  separate/%s done" % fname)

    emit("\n%-12s %7s %7s %7s %7s" % ("mutation", "MT-raw", "MT-bc", "sep-raw", "sep-bc"))
    def g(k, m):
        return ("%.2f" % C[k][m]) if m in C[k] else "  -"
    for m in MUTS:
        emit("%-12s %7s %7s %7s %7s" % (m[:12], g(("MT", "raw"), m), g(("MT", "bc"), m), g(("sep", "raw"), m), g(("sep", "bc"), m)))
    emit("\n%-12s %7.3f %7.3f %7.3f %7.3f" % ("MEAN",
         np.mean(list(C[("MT", "raw")].values())), np.mean(list(C[("MT", "bc")].values())),
         np.mean(list(C[("sep", "raw")].values())), np.mean(list(C[("sep", "bc")].values()))))
    emit("(reference: best linear model ~0.82-0.84)")
    emit("\nNN ABLATION OK")
except Exception as e:
    emit("[FAILED] %s: %s" % (type(e).__name__, e))
    emit(traceback.format_exc())
