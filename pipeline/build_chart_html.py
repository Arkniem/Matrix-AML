#!/usr/bin/env python3
"""Merge runs/single_modality/auc_*.tsv -> a self-contained interactive HTML:
per-mutation bar charts (one bar per modality = standalone held-out AUC), with a model selector.
-> runs/single_modality/modality_charts.html
"""
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, "runs", "single_modality")
import pandas as pd

fs = sorted(glob.glob(os.path.join(RUN, "auc_*.tsv")))
df = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
MODORD = ["Composition", "RNA", "ADT", "Lipid", "Metabolite", "GRN", "LSC", "Cell-comm"]
MODELS = ["logL2", "logL1", "elastic", "linSVM", "shrLDA", "PLS", "RF", "HistGB", "NaiveB", "kNN", "MLP"]
mods = [m for m in MODORD if m in set(df.modality)]
muts = sorted(df.mutation.unique())
npos = df.groupby("mutation").npos.max().to_dict()
data = {}
for mut in muts:
    sub = df[df.mutation == mut]
    data[mut] = {}
    for mo in MODELS:
        row = sub[sub.model == mo]
        data[mut][mo] = {r.modality: (None if pd.isna(r.auc) else round(float(r.auc), 3)) for r in row.itertuples()}
PAYLOAD = {"modalities": mods, "models": MODELS,
           "mutations": [{"name": m, "npos": int(npos.get(m, 0))} for m in muts], "data": data}

COLORS = {"Composition": "#4e79a7", "RNA": "#59a14f", "ADT": "#e15759", "Lipid": "#76b7b2",
          "Metabolite": "#edc948", "GRN": "#b07aa1", "LSC": "#ff9da7", "Cell-comm": "#f28e2b"}

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Modality contribution per mutation</title>
<style>
:root{--bg:#0f0f10;--panel:#1a1a1c;--line:#2a2a2e;--txt:#e8e8e6;--mut:#9a9a96;--accent:#5aa9c9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:22px 26px 60px}
h1{font-size:19px;font-weight:500;margin:0 0 3px}
.sub{color:var(--mut);font-size:13px;margin:0 0 18px;max-width:760px}
.ctrls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 14px}
.ctrls .lab{color:var(--mut);font-size:12px;margin-right:4px}
button{font:inherit;font-size:12px;padding:4px 11px;border-radius:7px;cursor:pointer;background:transparent;color:var(--mut);border:1px solid var(--line);transition:.12s}
button:hover{border-color:var(--accent);color:var(--txt)}
button.on{background:var(--accent);color:#06222c;border-color:var(--accent);font-weight:500}
.legend{display:flex;flex-wrap:wrap;gap:9px 16px;margin:0 0 18px;font-size:12px;color:var(--mut)}
.legend span{display:flex;align-items:center;gap:5px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:11px 13px}
.phead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px}
.pname{font-size:14px;font-weight:500}
.ppos{font-size:11px;color:var(--mut)}
.row{display:flex;align-items:center;gap:7px;margin:3px 0}
.rlab{width:80px;font-size:11px;color:var(--mut);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.track{position:relative;flex:1;height:13px;background:#101012;border:1px solid var(--line);border-radius:3px;overflow:hidden}
.fill{position:absolute;left:0;top:0;height:100%;border-radius:3px 0 0 3px}
.chance{position:absolute;top:-2px;height:17px;width:1px;background:#55555a}
.rval{width:30px;font-size:11px;text-align:right;font-variant-numeric:tabular-nums}
.foot{margin-top:22px;color:var(--mut);font-size:12px;max-width:760px}
</style></head><body>
<h1>Modality contribution per mutation</h1>
<p class="sub">Each bar is one modality used <b>alone</b> to predict that mutation on the sealed held-out set. Bar length = held-out AUC on a 0.4&ndash;1.0 scale; the tick marks 0.5 (chance). Switch models to see how the contribution changes.</p>
<div class="ctrls" id="ctrls"><span class="lab">model</span></div>
<div class="legend" id="legend"></div>
<div class="grid" id="grid"></div>
<p class="foot" id="foot"></p>
<script>
const D=__PAYLOAD__, COL=__COLORS__;
const LO=0.4,HI=1.0,CH=0.5; let model="Best";
const pct=v=>Math.max(0,Math.min(100,(v-LO)/(HI-LO)*100));
const clean=n=>n.replace("mut_","").replace("cyto_","");
function val(mut,mod){if(model==="Best"){let b=null;for(const m of D.models){const v=D.data[mut][m][mod];if(v!=null&&(b==null||v>b))b=v;}return b;}return D.data[mut][model][mod];}
function ctrls(){const c=document.getElementById("ctrls");c.querySelectorAll("button").forEach(b=>b.remove());
["Best",...D.models].forEach(m=>{const b=document.createElement("button");b.textContent=m;if(m===model)b.className="on";b.onclick=()=>{model=m;render();};c.appendChild(b);});}
function legend(){document.getElementById("legend").innerHTML=D.modalities.map(m=>`<span><span class="sw" style="background:${COL[m]}"></span>${m}</span>`).join("");}
function render(){ctrls();
document.getElementById("grid").innerHTML=D.mutations.map(mu=>{
 const rows=D.modalities.map(mod=>{const v=val(mu.name,mod);const w=v==null?0:pct(v);const lbl=v==null?"&mdash;":v.toFixed(2);
  return `<div class="row"><span class="rlab">${mod}</span><div class="track"><div class="fill" style="width:${w}%;background:${COL[mod]}"></div><div class="chance" style="left:${pct(CH)}%"></div></div><span class="rval">${lbl}</span></div>`;}).join("");
 return `<div class="panel"><div class="phead"><span class="pname">${clean(mu.name)}</span><span class="ppos">${mu.npos} pos</span></div>${rows}</div>`;}).join("");
document.getElementById("foot").textContent="Model = "+model+(model==="Best"?" (per-cell maximum across the 11 models — an optimistic ceiling).":".")+"  "+D.modalities.length+" of 8 modalities shown; "+D.mutations.length+" mutations on the sealed held-out (26 testable samples).";}
legend();render();
</script></body></html>"""

out = HTML.replace("__PAYLOAD__", json.dumps(PAYLOAD)).replace("__COLORS__", json.dumps(COLORS))
fp = os.path.join(RUN, "modality_charts.html")
with open(fp, "w", encoding="utf-8") as f:
    f.write(out)
print("modalities:", mods)
print("wrote", fp, "(%d KB)" % (len(out) // 1024))
