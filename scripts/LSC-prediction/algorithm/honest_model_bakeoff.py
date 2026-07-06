#!/usr/bin/env python3
"""
Honest, paired nested-CV bake-off of LSC-subtype model families on the 27
labeled training samples. Every feature-selection step is done INSIDE each
outer-train fold so the reported CV is an unbiased estimate of generalization.
Same repeated-stratified folds are reused across all models (paired comparison).
Metric: project WeightedObjective (0.45 p-LSC F1 + 0.45 m-LSC F1 + 0.10 p+m-LSC F1),
plus BalancedAccuracy and MacroF1.
"""
import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/saljh8/Dropbox/Collaborations/Grimes/UDON")
import LSC_classifier3 as v3  # reuse exact aggregate-model functions

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

TRAIN = "/Users/saljh8/Dropbox/Collaborations/Grimes/UDON/iterative_logistic_results/sample_cell_frequencies.tsv"
MIN_NONZERO = 2
N_REPEATS = 30
N_SPLITS = 3

df = pd.read_csv(TRAIN, sep="\t")
state_cols = [c for c in df.columns if c not in ("SampleID", "Class", "TotalCells")]
y_all = df["Class"].astype(str).values
print(f"[INFO] {len(df)} samples, {len(state_cols)} state columns")

def wobj(y_true, y_pred):
    f = {c: f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)
         for c in ["p-LSC", "m-LSC", "p+m-LSC"]}
    return 0.45*f["p-LSC"] + 0.45*f["m-LSC"] + 0.10*f["p+m-LSC"]

def metrics(y_true, y_pred):
    return dict(
        WeightedObjective=wobj(y_true, y_pred),
        BalancedAccuracy=balanced_accuracy_score(y_true, y_pred),
        MacroF1=f1_score(y_true, y_pred, average="macro", zero_division=0),
    )

# ---------- model arms: each returns y_pred for the outer-test fold ----------

def candidate_states(train_df):
    out = []
    for s in state_cols:
        if int(np.sum(train_df[s].astype(float).values > 0)) >= MIN_NONZERO:
            out.append(s)
    return out

def arm_aggregate(train_df, test_df, kp, km, seed):
    """v3's exact aggregate-lineage logistic, selection nested in train."""
    cand = candidate_states(train_df)
    pr = v3.one_vs_rest_state_ranking(train_df, cand, "p-LSC")
    mr = v3.one_vs_rest_state_ranking(train_df, cand, "m-LSC")
    p_states = pr["CellState"].tolist()[:kp]
    m_states = mr["CellState"].tolist()[:km]
    tr = v3.build_aggregate_features(train_df, p_states, m_states)
    te = v3.build_aggregate_features(test_df, p_states, m_states)
    feat = [c for c in tr.columns if c not in ("SampleID", "Class")]
    pipe = v3.get_logistic_pipeline("balanced", seed)
    pipe.fit(tr[feat].astype(float).values, tr["Class"].astype(str).values)
    classes = list(pipe.named_steps["clf"].classes_)
    thr = v3.compute_thresholds_from_training_probabilities(
        pipe.predict_proba(tr[feat].astype(float).values), tr["Class"].astype(str).values, classes)
    pp = pipe.predict_proba(te[feat].astype(float).values)
    return v3.apply_thresholded_prediction(pp, classes, thr)

def _logit():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, solver="lbfgs",
                      multi_class="multinomial", class_weight="balanced"))])

def _inner_obj(train_df, feats, seed):
    """pooled inner-CV WeightedObjective for a logistic on raw state freqs over `feats`."""
    X = train_df[feats].astype(float).values
    y = train_df["Class"].astype(str).values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tri, tei in skf.split(X, y):
        m = _logit().fit(X[tri], y[tri])
        yt.extend(y[tei]); yp.extend(m.predict(X[tei]))
    return wobj(np.array(yt), np.array(yp))

def arm_forward(train_df, test_df, max_k, seed):
    """Honest forward selection on raw state freqs, nested inside the train fold."""
    cand = candidate_states(train_df)
    chosen, best_obj, best_set = [], -1.0, []
    remaining = list(cand)
    while remaining and len(chosen) < max_k:
        scored = [(s, _inner_obj(train_df, chosen+[s], seed)) for s in remaining]
        s_best, o_best = max(scored, key=lambda t: t[1])
        chosen.append(s_best); remaining.remove(s_best)
        if o_best > best_obj + 1e-9:
            best_obj, best_set = o_best, list(chosen)
    feats = best_set if best_set else chosen[:1]
    X = train_df[feats].astype(float).values
    y = train_df["Class"].astype(str).values
    pipe = _logit().fit(X, y)
    classes = list(pipe.named_steps["clf"].classes_)
    thr = v3.compute_thresholds_from_training_probabilities(pipe.predict_proba(X), y, classes)
    pp = pipe.predict_proba(test_df[feats].astype(float).values)
    return v3.apply_thresholded_prediction(pp, classes, thr)

def arm_logit_all(train_df, test_df, seed):
    feats = candidate_states(train_df)
    pipe = _logit().fit(train_df[feats].astype(float).values, train_df["Class"].astype(str).values)
    return pipe.predict(test_df[feats].astype(float).values)

def arm_rf_all(train_df, test_df, seed):
    feats = candidate_states(train_df)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                random_state=seed)
    rf.fit(train_df[feats].astype(float).values, train_df["Class"].astype(str).values)
    return rf.predict(test_df[feats].astype(float).values)

MODELS = {
    "AGG-17/20 (v3 deployed)": lambda tr, te, s: arm_aggregate(tr, te, 17, 20, s),
    "AGG-6/6 (regularized)":   lambda tr, te, s: arm_aggregate(tr, te, 6, 6, s),
    "FWD-logit (<=4 states)":  lambda tr, te, s: arm_forward(tr, te, 4, s),
    "FWD-logit (<=6 states)":  lambda tr, te, s: arm_forward(tr, te, 6, s),
    "LOGIT-all-states":        lambda tr, te, s: arm_logit_all(tr, te, s),
    "RF-all-states":           lambda tr, te, s: arm_rf_all(tr, te, s),
}

# ---------- paired repeated stratified CV ----------
records = {name: [] for name in MODELS}
for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rep)
    pooled = {name: ([], []) for name in MODELS}
    for tri, tei in skf.split(df, y_all):
        tr = df.iloc[tri].reset_index(drop=True)
        te = df.iloc[tei].reset_index(drop=True)
        yte = te["Class"].astype(str).values
        for name, fn in MODELS.items():
            yp = fn(tr, te, rep)
            pooled[name][0].extend(yte); pooled[name][1].extend(yp)
    for name in MODELS:
        yt, yp = pooled[name]
        records[name].append(metrics(np.array(yt), np.array(yp)))
    print(f"[INFO] repeat {rep+1}/{N_REPEATS} done")

rows = []
for name in MODELS:
    rdf = pd.DataFrame(records[name])
    rows.append(dict(
        Model=name,
        WeightedObjective_mean=rdf.WeightedObjective.mean(),
        WeightedObjective_std=rdf.WeightedObjective.std(),
        BalancedAccuracy_mean=rdf.BalancedAccuracy.mean(),
        MacroF1_mean=rdf.MacroF1.mean(),
    ))
res = pd.DataFrame(rows).sort_values("WeightedObjective_mean", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 160, "display.max_columns", 20)
print("\n================ HONEST NESTED-CV BAKE-OFF ({}x{}-fold, paired) ================".format(N_REPEATS, N_SPLITS))
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
out = "/Users/saljh8/Dropbox/Collaborations/Grimes/UDON/deployed_results_combined_annotations_sample_level/honest_model_bakeoff_results.tsv"
res.to_csv(out, sep="\t", index=False)
print("\nWrote:", out)
print("\nWINNER (honest WeightedObjective):", res.iloc[0]["Model"])
