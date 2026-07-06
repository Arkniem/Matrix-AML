"""Model zoo — fold-safe sklearn Pipelines behind a uniform interface.

Every model is a Pipeline so that ALL preprocessing (scaling, etc.) is fit inside
each CV fold, never on held-out data. `build(names)` returns
    name -> (estimator: Pipeline, param_grid: dict)
ready for GridSearchCV. Add a model here (a new key) to expand the panel an agent
can choose from via DecisionHooks.choose_models.
"""
from __future__ import annotations
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 0
# Respect the LSF slot allocation: the job script sets AMLMM_NJOBS to the reserved
# slots so RF / GridSearchCV don't oversubscribe a shared node. -1 (all cores) locally.
NJOBS = int(os.environ.get("AMLMM_NJOBS", "-1"))


def _rf():
    est = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=NJOBS)),
    ])
    grid = {"clf__max_depth": [None, 8]}
    return est, grid


def _logreg():
    est = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    grid = {"clf__C": [0.1, 1.0]}
    return est, grid


def _gboost():
    est = Pipeline([
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=300, random_state=RANDOM_STATE)),
    ])
    grid = {"clf__max_depth": [None, 4]}
    return est, grid


MODEL_ZOO = {"rf": _rf, "logreg": _logreg, "gboost": _gboost}


def build(names) -> dict:
    out = {}
    for n in names:
        if n not in MODEL_ZOO:
            raise KeyError(f"unknown model '{n}'; available: {list(MODEL_ZOO)}")
        out[n] = MODEL_ZOO[n]()
    return out


def simple_estimator():
    """A single cheap pipeline used for the permutation/chance baseline."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
