"""AAMI-style per-class metrics (Se = recall, +P = precision, F1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import CLASSES


def report(y_true, y_pred) -> dict:
    labels = list(range(len(CLASSES)))
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(np.mean(f)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    for i, c in enumerate(CLASSES):
        out[c] = dict(se=float(r[i]), ppv=float(p[i]), f1=float(f[i]),
                      support=int(sup[i]))
    return out


def format_report(rep: dict, title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"  accuracy {rep['accuracy']*100:6.2f}%   macro-F1 {rep['macro_f1']:.3f}")
    for c in CLASSES:
        d = rep[c]
        lines.append(f"  {c}: Se {d['se']:.3f}  +P {d['ppv']:.3f}  "
                     f"F1 {d['f1']:.3f}  (n={d['support']})")
    return "\n".join(lines)
