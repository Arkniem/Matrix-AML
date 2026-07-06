"""Deploy agent — per-specimen inference using the Discovery training corpus.

Discovery certified WHICH (field x modality x cell-state) channels carry signal and HOW MUCH (the
permutation-calibrated `weight`). Deploy makes the actual per-specimen call by **nearest-neighbor
label-transfer** within those certified channels: for each channel, match the specimen's pseudobulk to
the TRAINING pseudobulks of the SAME cell-state (Spearman rank correlation on the channel's stored
`feature_space`), vote the neighbors' held-out true labels, and combine channels weighted by the
Discovery weight. A channel Discovery could not predict above chance (weight 0) is never consulted.

Why NN (not the refit RF)? The plan's confirmed primitive is signature matching, it is model-free and
auditable ("your GMP cells most resemble these NPM1+ training samples"), and Discovery's weight already
encodes how much to trust each channel. Spearman is rank-based -> invariant to Discovery's cp10k/log1p,
so the transfer is valid without replicating the in-fold normalization.

v1 channels need NO modality imputation (directly available from a specimen's scRNA):
  * composition  — sample-level cell-state fractions (the `comp::<state>` feature space),
  * RNA          — the specimen's per-cell-state pseudobulks,
  * UDON         — the specimen's RNA pseudobulk vs the control-normalized `udon_result.h5ad`,
                   restricted to MarkerFinder markers + the same cell-state (a SEPARATE descriptive
                   signature channel: it was not in the Discovery sweep, so it carries no certified
                   weight and is reported alongside the certified call, not folded into it).
Imputed modalities (ADT/Metabolite/GRN/Lipid) are DEFERRED to v2 (need the rna2* bundles) — same
deferral as `ingest_patient.py`. The Arbiter (next agent) reconciles the certified call + UDON evidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field as _dc_field

import numpy as np
import pandas as pd

from . import discovery as D, pseudobulk_io as pio


@dataclass
class DeployConfig:
    k: int = 5                                    # neighbors per channel
    min_weight: float = 0.05                      # only consult channels Discovery weighted above this
    modalities: tuple = ("composition", "RNA")    # v1 no-imputation channels (ADT/Metabolite -> v2)
    max_combos_per_field: int = 8                 # cap channels/field (top-weighted) for tractability
    use_udon: bool = True
    udon_k: int = 5
    udon_cell_states_top: int = 6                 # specimen cell-states (by n_cells) to run the UDON signature on


# --------------------------------------------------------------------------- the NN-transfer primitive
def nn_vote(query, ref_values, labels, ids, k):
    """Top-k Spearman-nearest training rows -> a positive-correlation-weighted vote over their labels.
    Returns (probs:{label:frac}, neighbors:[{id,corr,label}]). Falls back to a uniform top-k vote if no
    neighbor is positively correlated (degenerate query)."""
    corr = pio._spearman_to_each(query, ref_values)
    order = np.argsort(-corr)[:int(k)]
    votes, wsum, neigh = {}, 0.0, []
    for j in order:
        c = float(corr[j]); w = max(c, 0.0); lab = str(labels[j])
        votes[lab] = votes.get(lab, 0.0) + w; wsum += w
        neigh.append({"id": str(ids[j]), "corr": round(c, 3), "label": lab})
    if wsum <= 0.0:                                # all neighbors anti-/zero-correlated -> uniform top-k
        for j in order:
            votes[str(labels[j])] = votes.get(str(labels[j]), 0.0) + 1.0
        wsum = float(len(order)) or 1.0
    probs = {c: round(v / wsum, 4) for c, v in votes.items()}
    return probs, neigh


# --------------------------------------------------------------------------- reference + query builders
def _ids_to_donor(ctx, ids, modality) -> dict:
    """Map reference row ids -> donor_group (for honest leave-one-donor-out in the self-test)."""
    s = ctx.tables["samples"]
    if modality == "composition":                 # composition association id == sample_key
        return {i: str(s["donor_group"].get(i, "?")) for i in ids}
    pb = ctx.tables["pseudobulks"]                 # pseudobulk id -> sample_key -> donor
    out = {}
    for i in ids:
        sk = pb["sample_key"].get(i)
        out[i] = str(s["donor_group"].get(sk, "?")) if sk is not None else "?"
    return out


def _query_vec(ctx, modality, cell_state, sample_key, feature_space):
    """The specimen's vector for one channel, aligned to the channel's feature_space (missing -> 0).
    Built with the SAME readers Discovery used, so query and reference live in the same space."""
    if modality == "composition":
        X = D._sample_level_matrix(ctx, "composition", {sample_key})
    else:
        X = pio.cellstate_modality_matrix(ctx, modality, cell_state, sample_keys={sample_key})
    if X.shape[0] == 0:
        return None
    return X.iloc[0].reindex(feature_space).fillna(0.0).values.astype(float)


def _ref_frame(ctx, dr, field, modality, cell_state, feature_space):
    """Reference matrix (training pseudobulks of this channel) + their held-out true labels, from the
    Discovery associations (ids + true) re-hydrated to features via the same readers, restricted to the
    channel's feature_space. Returns (DataFrame[id x feature], Series id->label) or (None, None)."""
    A = dr.associations_for(field, modality, cell_state)
    if A.empty:
        return None, None
    labels = dict(zip(A["pseudobulk_id"].astype(str), A["true"].astype(str)))
    ids = list(labels)
    if modality == "composition":
        X = D._sample_level_matrix(ctx, "composition", set(ids))
    else:
        X = pio.cellstate_modality_matrix(ctx, modality, cell_state)   # all of the cell-state (cached)
    X.index = X.index.astype(str)
    X = X.reindex([i for i in ids if i in set(X.index)])
    if X.shape[0] == 0:
        return None, None
    X = X.reindex(columns=feature_space).fillna(0.0)
    lab = pd.Series({i: labels[i] for i in X.index})
    return X, lab


# --------------------------------------------------------------------------- one certified channel
def deploy_combo(ctx, dr, field, modality, cell_state, sample_key, k, exclude_donor=None):
    """NN label-transfer for one certified (field x modality x cell-state) channel. `exclude_donor`
    drops all reference rows of that donor (+ the query itself) for an honest self-test; pass None for a
    genuinely external specimen. Returns a channel dict or None (no feature space / query absent / <2 refs)."""
    fs = dr.feature_space(field, modality, cell_state)
    if not fs:
        return None
    q = _query_vec(ctx, modality, cell_state, sample_key, fs)
    if q is None:
        return None
    refX, lab = _ref_frame(ctx, dr, field, modality, cell_state, fs)
    if refX is None:
        return None
    drop = {str(sample_key)}
    if exclude_donor is not None:
        donors = _ids_to_donor(ctx, list(refX.index), modality)
        drop |= {i for i, d in donors.items() if d == str(exclude_donor)}
    keep = [i for i in refX.index if i not in drop]
    if len(keep) < 2:
        return None
    refX = refX.loc[keep]; lab = lab.loc[keep]
    probs, neigh = nn_vote(q, refX.values, lab.values, list(refX.index), min(k, len(keep)))
    pred = max(probs, key=probs.get)
    return {"field": field, "modality": modality, "cell_state": cell_state,
            "weight": dr.weight(field, modality, cell_state), "pred": pred, "prob": probs[pred],
            "probs": probs, "n_ref": len(keep), "neighbors": neigh}


def _certified_combos(dr, field, config):
    """Top-weighted certified channels for a field among the v1 modalities (weight>min_weight + has a
    stored feature_space). Returns [(modality, cell_state, weight)] sorted by weight desc, capped."""
    out = []
    for e in dr.association_index.values():
        if e.get("field") != field or e.get("modality") not in config.modalities:
            continue
        w = e.get("weight") or 0.0
        if w > config.min_weight and e.get("feature_space"):
            out.append((e["modality"], e["cell_state"], float(w)))
    out.sort(key=lambda t: -t[2])
    return out[:config.max_combos_per_field] if config.max_combos_per_field else out


# --------------------------------------------------------------------------- certified call for a field
def deploy_field(ctx, dr, field, sample_key, config=None, exclude_donor=None):
    """The certified per-specimen call for one field: weight-moderated combination of the certified
    channels' NN votes. Returns leading label + confidence + class probabilities + per-channel detail."""
    config = config or DeployConfig()
    combos = _certified_combos(dr, field, config)
    channels, agg = [], {}
    for modality, cell_state, w in combos:
        r = deploy_combo(ctx, dr, field, modality, cell_state, sample_key, config.k, exclude_donor)
        if r is None:
            continue
        r["weight"] = w
        channels.append(r)
        for c, p in r["probs"].items():
            agg[c] = agg.get(c, 0.0) + w * p
    if not agg:
        return {"field": field, "status": "no_channels",
                "reason": "no certified channel available for this specimen"}
    tot = sum(agg.values()) or 1.0
    final = {c: round(v / tot, 4) for c, v in agg.items()}
    ranked = sorted(final.items(), key=lambda kv: -kv[1])
    leading, conf = ranked[0]
    margin = round(conf - (ranked[1][1] if len(ranked) > 1 else 0.0), 4)
    return {"field": field, "status": "ok", "leading": leading, "confidence": conf, "margin": margin,
            "class_probabilities": final, "n_channels": len(channels),
            "channels": [{"modality": c["modality"], "cell_state": c["cell_state"],
                          "weight": round(c["weight"], 3), "pred": c["pred"], "prob": c["prob"],
                          "n_ref": c["n_ref"]} for c in channels],
            "channel_detail": channels}


# --------------------------------------------------------------------------- UDON signature (descriptive)
def udon_subtype_signature(ctx, sample_key, config=None):
    """Per-cell-state UDON signature match: the specimen's RNA pseudobulk vs the control-normalized
    `udon_result.h5ad`, restricted to MarkerFinder markers + the same cell-state (Spearman). Reports the
    top-k neighbors' canonicalized subtype (Annotation) per cell-state + a cross-state consensus. This is
    DESCRIPTIVE evidence (no Discovery weight) — the Arbiter weighs it separately from the certified call."""
    config = config or DeployConfig()
    markers = pio.udon_markers(ctx, "RNA")
    if not markers:
        return {"status": "unavailable", "reason": "no MarkerFinder markers / udon_result.h5ad"}
    cmap = ctx.hooks.canonical_label_map([])      # base map; applied per-label below
    states = _specimen_cell_states(ctx, sample_key, config.udon_cell_states_top)
    per_state, consensus = [], {}
    for C in states:
        refX, obs = pio.udon_result_matrix(ctx, "RNA", markers=markers, cell_state=C)
        if refX is None or refX.shape[0] < 2 or refX.shape[1] == 0:
            continue
        q = _query_vec(ctx, "RNA", C, sample_key, list(refX.columns))
        if q is None:
            continue
        raw = obs.reindex(refX.index)["Annotation"].astype(str) if "Annotation" in obs.columns else None
        if raw is None:
            continue
        canon = raw.map(lambda a: ctx.hooks.canonical_label_map([a]).get(a, a))
        probs, neigh = nn_vote(q, refX.values, canon.values, list(refX.index), min(config.udon_k, refX.shape[0]))
        for n, lab in zip(neigh, [canon.get(n["id"], n["label"]) for n in neigh]):
            n["subtype"] = lab
        call = max(probs, key=probs.get)
        per_state.append({"cell_state": C, "call": call, "prob": probs[call],
                          "neighbors": neigh, "n_ref": int(refX.shape[0])})
        for c, p in probs.items():
            consensus[c] = consensus.get(c, 0.0) + p
    if not per_state:
        return {"status": "no_match", "reason": "no cell-state matched the UDON signature object"}
    tot = sum(consensus.values()) or 1.0
    cons = {c: round(v / tot, 4) for c, v in consensus.items()}
    lead = max(cons, key=cons.get)
    return {"status": "ok", "consensus_call": lead, "consensus_prob": cons[lead],
            "consensus_distribution": cons, "n_cell_states": len(per_state), "per_state": per_state}


def _specimen_cell_states(ctx, sample_key, top):
    """The specimen's cell-states ranked by n_cells (most-represented first)."""
    pb = ctx.tables["pseudobulks"]
    mine = pb[pb["sample_key"].astype(str) == str(sample_key)]
    if "n_cells" in mine.columns:
        mine = mine.sort_values("n_cells", ascending=False)
    return list(mine["cell_state"].astype(str))[:int(top)]


# --------------------------------------------------------------------------- whole-specimen assembly
def run_deploy_atlas(ctx, dr, sample_key, fields, config=None, exclude_donor="auto"):
    """Run Deploy for an ATLAS sample (the self-test / leave-one-donor-out path): query = an existing
    cohort sample, references = the rest (its donor dropped when exclude_donor='auto'). Returns a deploy
    report (certified calls per field + the UDON signature). For a genuinely external specimen, the
    scRNA front-end (v2) builds the same query vectors and calls deploy_field/udon with exclude_donor=None."""
    config = config or DeployConfig()
    if exclude_donor == "auto":
        exclude_donor = str(ctx.tables["samples"]["donor_group"].get(sample_key, None))
    calls = {f: deploy_field(ctx, dr, f, sample_key, config, exclude_donor=exclude_donor) for f in fields}
    udon = udon_subtype_signature(ctx, sample_key, config) if config.use_udon else {"status": "off"}
    return {"mode": "deploy_atlas_selftest", "sample_key": sample_key, "exclude_donor": exclude_donor,
            "discovery_run": dr.run_dir, "certified_calls": calls, "udon_signature": udon}
