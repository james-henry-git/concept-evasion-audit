"""
Probe training and evaluation.

Experiment A: fixed single-layer linear probe (their method)
Experiment B: CAZ-aware probe at concept-specific depth
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import pickle


def train_linear_probe(
    hidden_states: np.ndarray,  # (N, d)
    labels: np.ndarray,         # (N,) binary
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
    clf.fit(hidden_states, labels)
    return clf


def recall_at_fpr(probe, hidden_states: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01) -> float:
    """True positive rate at a fixed false positive rate threshold."""
    scores = probe.predict_proba(hidden_states)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find the threshold where FPR is closest to target_fpr from below
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def auroc(probe, hidden_states: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    scores = probe.predict_proba(hidden_states)[:, 1]
    return float(roc_auc_score(labels, scores))


def save_probe(probe: LogisticRegression, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(probe, f)


def load_probe(path: Path) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_probe(probe, hidden_states: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "auroc": auroc(probe, hidden_states, labels),
        "recall_at_1pct_fpr": recall_at_fpr(probe, hidden_states, labels, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(probe, hidden_states, labels, 0.05),
    }
