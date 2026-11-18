"""Risk-model metrics: AUC/Gini, KS, and Population Stability Index (PSI)."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y_true, score):
    return 2 * roc_auc_score(y_true, score) - 1


def ks_statistic(y_true, score):
    """Max gap between cumulative default and non-default distributions."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins=10):
    """Population Stability Index between a reference and a new sample.

    < 0.10  stable | 0.10-0.25  watch | > 0.25  significant shift
    Bin edges come from the reference (expected) sample's quantiles.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant / few-valued feature: use its distinct values
        edges = np.unique(np.concatenate([edges, np.unique(expected)]))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value):
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "watch"
    return "ALERT"
