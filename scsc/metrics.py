"""Clustering metrics: unsupervised ACC (Hungarian), NMI, ARI, F-score."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
)


def cluster_acc(y_true, y_pred):
    """Best label permutation accuracy via the Hungarian algorithm."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    assert y_true.size == y_pred.size
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        w[p, t] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return float(w[row, col].sum()) / y_pred.size


def clustering_metrics(y_true, y_pred):
    return {
        "ACC": cluster_acc(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "F": float(fowlkes_mallows_score(y_true, y_pred)),
    }
