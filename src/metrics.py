"""Imbalance-aware metrics. The positive class is always deceptive (label 1)."""
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

METRICS = ["macro_f1", "precision_deceptive", "recall_deceptive", "f1_deceptive", "auc_roc"]


def evaluate(y_true, y_pred, y_score):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_deceptive": p[0],
        "recall_deceptive": r[0],
        "f1_deceptive": f[0],
        "auc_roc": roc_auc_score(y_true, y_score),
    }
