"""Torch-free baselines (gradient boosting) for fast ceiling / sweep numbers.

Modes
  dense_raw  physical-mV beat window straight from the dataset (no AFE)
  dense_afe  beat window after the analog HPF/LPF (what the comparators see)
  spike      hardware-exact spike histograms + leaky-integrator channel
All modes also receive the causal RR / spike-energy scalars.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .metrics import report


def featurize(ds: dict, mode: str, ncol: int = 13) -> np.ndarray:
    scal = ds["scal"][:, :ncol].astype(np.float32)
    if mode == "spike":
        body = np.concatenate([ds["hist"].reshape(len(ds["y"]), -1),
                               ds["res"].reshape(len(ds["y"]), -1)], 1).astype(np.float32)
    elif mode == "dense_raw":
        body = ds["raw"]
    elif mode == "dense_afe":
        body = ds["filt"]
    else:
        raise ValueError(mode)
    return np.concatenate([body, scal], axis=1)


def fit_eval(train: dict, test: dict, mode: str, max_iter: int = 200,
             seed: int = 0) -> dict:
    Xtr, ytr = featurize(train, mode), train["y"].astype(int)
    Xte, yte = featurize(test, mode), test["y"].astype(int)
    # sqrt of the balanced weights: helps minority classes without letting the
    # tiny F class dominate the loss.
    w = np.sqrt(compute_sample_weight("balanced", ytr))
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed)
    clf.fit(Xtr, ytr, sample_weight=w)
    rep = report(yte, clf.predict(Xte))
    rep["idle_sparsity"] = test["stats"]["idle_sparsity"]
    rep["spikes_per_s"] = test["stats"]["spikes_per_s"]
    return rep
