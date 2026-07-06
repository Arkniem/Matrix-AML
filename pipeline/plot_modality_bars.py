#!/usr/bin/env python3
"""Grouped-bar figure: one subplot per mutation, x-axis grouped by model, one bar per modality.
Reads runs/single_modality/auc_*.tsv (whatever modalities are done). -> modality_grouped_bars.png
"""
import os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(os.path.dirname(HERE), "runs", "single_modality")
df = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(os.path.join(RUN, "auc_*.tsv")))],
               ignore_index=True)

MODORD = ["Composition", "RNA", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
MODELS = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]
COLORS = {"Composition": "#4e79a7", "RNA": "#59a14f", "ADT": "#e15759", "Lipid": "#76b7b2",
          "Metabolite": "#edc948", "GRN": "#b07aa1", "LSC": "#ff9da7", "Cell-comm": "#f28e2b"}
mods = [m for m in MODORD if m in set(df.modality)]
muts = sorted(df.mutation.unique())
npos = df.groupby("mutation").npos.max().to_dict()
clean = lambda n: n.replace("mut_", "").replace("cyto_", "")
lut = {(r.mutation, r.model, r.modality): (np.nan if pd.isna(r.auc) else r.auc) for r in df.itertuples()}

ncol = 2
nrow = int(np.ceil(len(muts) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.7 * nrow))
axes = np.array(axes).reshape(-1)
x = np.arange(len(MODELS))
nb = len(mods)
width = 0.82 / nb

for i, mut in enumerate(muts):
    ax = axes[i]
    for j, mod in enumerate(mods):
        vals = [lut.get((mut, mo, mod), np.nan) for mo in MODELS]
        ax.bar(x + (j - (nb - 1) / 2) * width, vals, width, color=COLORS[mod],
               edgecolor="white", linewidth=0.2)
    ax.axhline(0.5, ls="--", lw=0.8, color="#888")
    ax.set_title("%s  (%d pos)" % (clean(mut), int(npos.get(mut, 0))), fontsize=10, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7)
    ax.set_ylabel("AUC", fontsize=8)
    ax.grid(axis="y", lw=0.4, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

for k in range(len(muts), len(axes)):
    axes[k].set_visible(False)

handles = [Patch(facecolor=COLORS[m], label=m) for m in mods]
fig.legend(handles=handles, loc="upper center", ncol=len(mods), fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Single-modality held-out AUC per mutation  —  groups = models, bars = modalities  (dashed = 0.5 chance)",
             fontsize=13, y=1.012)
fig.tight_layout(rect=[0, 0, 1, 0.99])
out = os.path.join(RUN, "modality_grouped_bars.png")
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print("modalities:", mods, "| mutations:", len(muts))
print("wrote", out)
