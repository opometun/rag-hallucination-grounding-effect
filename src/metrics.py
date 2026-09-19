"""Prediction rule and F1 metrics for the hallucination detector.

Single source of truth for two things: how logits become classes, and how predictions and
labels become F1. Training, test evaluation and analysis all import from here so they
cannot drift apart on a tie or on how F1 is aggregated.
"""

import numpy as np

CLEAN = 0
HALLUCINATED = 1


def _to_numpy(x):
    # Torch tensors come through detach/cpu/numpy, anything else through asarray, so the
    # module never has to import torch.
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def predict_from_logits(logits):
    logits = _to_numpy(logits)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"expected logits of shape [n, 2], got {logits.shape}")
    # >= rather than argmax. argmax returns the first maximum, so an exact tie would come
    # out as class 0, and the declared policy is that a tie goes to the positive class.
    return (logits[:, HALLUCINATED] >= logits[:, CLEAN]).astype(np.int64)


def _as_arrays(preds, labels):
    preds, labels = _to_numpy(preds), _to_numpy(labels)
    if preds.shape != labels.shape:
        raise ValueError(
            f"preds and labels differ in shape, {preds.shape} against {labels.shape}"
        )
    return preds, labels


def _prf(preds, labels, positive):
    tp = int(((preds == positive) & (labels == positive)).sum())
    fp = int(((preds == positive) & (labels != positive)).sum())
    fn = int(((preds != positive) & (labels == positive)).sum())
    # Zero denominator means 0.0, including the case where a class has no predicted and no
    # actual members. Not 1.0, not nan, and nothing divides by zero.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def positive_f1(preds, labels):
    preds, labels = _as_arrays(preds, labels)
    return _prf(preds, labels, HALLUCINATED)[2]


def macro_f1(preds, labels):
    preds, labels = _as_arrays(preds, labels)
    per_class = [_prf(preds, labels, c)[2] for c in (CLEAN, HALLUCINATED)]
    return sum(per_class) / len(per_class)


def scores(preds, labels):
    """Everything the training loop wants in one pass over the accumulated arrays."""
    preds, labels = _as_arrays(preds, labels)
    precision, recall, f1 = _prf(preds, labels, HALLUCINATED)
    clean_f1 = _prf(preds, labels, CLEAN)[2]
    return {
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "clean_f1": clean_f1,
        "macro_f1": (clean_f1 + f1) / 2,
    }
