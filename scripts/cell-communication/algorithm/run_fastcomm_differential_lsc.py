#!/usr/bin/env python3
"""p-LSC vs m-LSC differential cell-cell communication on the per-sample fastComm results.

Faithfully replicates cellHarmony-web's cell-communication differential strategy
(cellHarmony/flask/pipeline.py::_run_cell_communication_differential):
  - reconstruct the per-sample fastComm long table (split=sample) from the
    per-sample h5ads;
  - pivot interaction_id (sender|receiver|ligand|receptor) x sample, fastcomm_score,
    aggfunc=max, fill 0; ensure every group sample is a column;
  - per interaction: case=group1 (p-LSC) samples, control=group2 (m-LSC) samples;
    delta = case_mean - control_mean; log2fc = log2((case+1e-4)/(control+1e-4));
    replicate_group_contrast (both groups >=2) -> Mann-Whitney U two-sided p;
  - BH-FDR; significant = fdr<=0.10 & |delta|>=0.05; sort by [fdr, |delta|, max_sample_score];
  - summary per receiver population.
Groups = LSC classifier PredictedClass (LSC_classification_summary.tsv), joined to
fastComm samples on (Sample, Dataset). case=p-LSC, control=m-LSC (positive = higher in p-LSC).

Outputs: cell_communication_fastcomm/differential_pLSC_vs_mLSC/
  DEG_detailed_cell_communication_pLSC_vs_mLSC.tsv  (== cell_communication_comparison_*)
  DEG_summary_cell_communication_pLSC_vs_mLSC.tsv , manifest.json , README.md
"""
import os, glob, json, time, datetime
import numpy as np, pandas as pd, anndata as ad
from scipy import stats

PB = "/Users/saljh8/Dropbox/Collaborations/Grimes/UDON/cellHarmony-datasets/final/pseudobulk"
CC = os.path.join(PB, "cell_communication_fastcomm")
PER = os.path.join(CC, "per_sample")
LSC_TSV = os.path.join(PB, "LSC-classification", "LSC_classification_summary.tsv")
OUT = os.path.join(CC, "differential_pLSC_vs_mLSC")
CASE, CONTROL, TAG = "p-LSC", "m-LSC", "pLSC_vs_mLSC"
SIG_FDR, SIG_DELTA = 0.10, 0.05


def _bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n); out[order] = ranked
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # LSC class per (Sample, Dataset)
    lsc = pd.read_csv(LSC_TSV, sep="\t")
    lsc["Sample"] = lsc["Sample"].astype(str); lsc["Dataset"] = lsc["Dataset"].astype(str)
    lsc_map = {(r.Sample, r.Dataset): str(r.PredictedClass) for r in lsc.itertuples()}

    # fastComm samples + their Dataset (from the combined matrix obs)
    C = ad.read_h5ad(os.path.join(CC, "combined_sample_by_interaction.h5ad"))
    ds_of = dict(zip(C.obs_names.astype(str), C.obs["Dataset"].astype(str)))
    cls_of = {s: lsc_map.get((s, ds_of.get(s, "")), None) for s in C.obs_names.astype(str)}
    g1 = sorted([s for s, c in cls_of.items() if c == CASE])      # p-LSC samples (case)
    g2 = sorted([s for s, c in cls_of.items() if c == CONTROL])   # m-LSC samples (control)
    print(f"fastComm samples: {C.n_obs} | {CASE} (case): {len(g1)} | {CONTROL} (control): {len(g2)}")
    wanted = set(g1) | set(g2)
    assert len(g1) >= 2 and len(g2) >= 2, "need >=2 samples per group for replicate_group_contrast"

    # reconstruct the per-sample fastComm long table for the two groups
    frames = []
    for h in sorted(glob.glob(os.path.join(PER, "*.h5ad"))):
        A = ad.read_h5ad(h)
        s = str(A.uns["sample"])
        if s not in wanted:
            continue
        df = A.obs[["sender_state", "receiver_state", "ligand", "receptor", "pathway", "fastcomm_score"]].copy()
        vn = list(A.var_names)
        Xden = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
        for mt in ("receiver_response_score", "lr_expression_score_scaled"):
            df[mt] = Xden[:, vn.index(mt)] if mt in vn else np.nan
        df["split"] = s
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    for c in ["sender_state", "receiver_state", "ligand", "receptor"]:
        long[c] = long[c].astype(str)
    long["fastcomm_score"] = pd.to_numeric(long["fastcomm_score"], errors="coerce").fillna(0.0)
    long["interaction_id"] = long[["sender_state", "receiver_state", "ligand", "receptor"]].agg("|".join, axis=1)
    print(f"long table rows: {len(long)} | interactions: {long['interaction_id'].nunique()}")

    # pivot interaction x sample (fastcomm_score), fill 0; ensure all group samples present
    M = long.pivot_table(index="interaction_id", columns="split", values="fastcomm_score", aggfunc="max", fill_value=0.0)
    for s in wanted:
        if s not in M.columns:
            M[s] = 0.0
    ann = long.sort_values("fastcomm_score", ascending=False).drop_duplicates("interaction_id", keep="first").set_index("interaction_id")

    g1c, g2c = list(g1), list(g2)
    case = M[g1c].to_numpy(float); control = M[g2c].to_numpy(float)
    case_mean = case.mean(1); control_mean = control.mean(1)
    delta = case_mean - control_mean
    log2fc = np.log2((case_mean + 1e-4) / (control_mean + 1e-4))
    pvals = np.ones(M.shape[0])
    for i in range(M.shape[0]):
        cv, kv = case[i], control[i]
        if np.unique(np.concatenate([cv, kv])).size > 1:
            try:
                pvals[i] = float(stats.mannwhitneyu(cv, kv, alternative="two-sided").pvalue)
            except Exception:
                pvals[i] = 1.0
    a = ann.loc[M.index]
    det = pd.DataFrame({
        "population": a["receiver_state"].values, "sender_state": a["sender_state"].values,
        "receiver_state": a["receiver_state"].values, "ligand": a["ligand"].values, "receptor": a["receptor"].values,
        "pathway": a["pathway"].values,
        "gene": a["sender_state"].astype(str).values + "->" + a["receiver_state"].astype(str).values + ":" +
                a["ligand"].astype(str).values + "->" + a["receptor"].astype(str).values,
        "interaction": a["ligand"].astype(str).values + "->" + a["receptor"].astype(str).values,
        "case_label": CASE, "control_label": CONTROL,
        "case_mean_score": case_mean, "control_mean_score": control_mean, "delta_score": delta, "log2fc": log2fc,
        "pval": pvals, "comparison_mode": "replicate_group_contrast",
        "n_case": len(g1c), "n_control": len(g2c),
        "max_sample_score": np.maximum(case.max(1), control.max(1)),
        "receiver_response_score": pd.to_numeric(a["receiver_response_score"], errors="coerce").values,
        "lr_expression_score": pd.to_numeric(a["lr_expression_score_scaled"], errors="coerce").values,
    }, index=M.index)
    det["abs_delta_score"] = det["delta_score"].abs()
    det["fdr"] = _bh_fdr(det["pval"].fillna(1.0).to_numpy())
    det = det.sort_values(["fdr", "abs_delta_score", "max_sample_score"], ascending=[True, False, False]).reset_index(drop=True)

    comp = os.path.join(OUT, f"cell_communication_comparison_{TAG}.tsv")
    det.to_csv(comp, sep="\t", index=False)
    det.to_csv(os.path.join(OUT, f"DEG_detailed_cell_communication_{TAG}.tsv"), sep="\t", index=False)
    det["significant"] = (det["fdr"] <= SIG_FDR) & (det["abs_delta_score"] >= SIG_DELTA)
    summ = det.groupby("population", as_index=False).agg(
        num_DEG=("significant", "sum"), tested_genes=("gene", "count"), max_abs_delta_score=("abs_delta_score", "max")
    ).sort_values("num_DEG", ascending=False)
    summ.to_csv(os.path.join(OUT, f"DEG_summary_cell_communication_{TAG}.tsv"), sep="\t", index=False)

    nsig = int(det["significant"].sum())
    up = int(((det["significant"]) & (det["delta_score"] > 0)).sum())
    dn = int(((det["significant"]) & (det["delta_score"] < 0)).sum())
    manifest = {"modality": "cell_communication", "comparison": TAG, "case_label": CASE, "control_label": CONTROL,
                "comparison_mode": "replicate_group_contrast", "n_case_samples": len(g1c), "n_control_samples": len(g2c),
                "case_samples": g1c, "control_samples": g2c,
                "method": "cellHarmony cell-communication differential (delta of per-sample fastComm scores, Mann-Whitney U, BH-FDR)",
                "significance": f"fdr<={SIG_FDR} & |delta_score|>={SIG_DELTA}",
                "n_interactions_tested": int(det.shape[0]), "n_significant": nsig,
                f"n_up_in_{CASE}": up, f"n_up_in_{CONTROL}": dn,
                "input": "per-sample fastComm results (cell_communication_fastcomm/per_sample/*.h5ad)"}
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"tested interactions: {det.shape[0]} | significant (fdr<=0.10 & |delta|>=0.05): {nsig} "
          f"({up} up in {CASE}, {dn} up in {CONTROL})  [{time.time()-t0:.0f}s]")
    print("top by significance:")
    print(det.head(12)[["gene", "pathway", "case_mean_score", "control_mean_score", "delta_score", "log2fc", "pval", "fdr"]].to_string(index=False))


if __name__ == "__main__":
    main()
