"""Decision-rule post-processing for macro-F1, fitted only on out-of-fold scores.

Why offsets: linearising macro-F1 around the optimal confusion matrix gives the
Bayes-optimal rule  h(x) = argmax_c (2 p_c(x) - F1_c) / (pi_c + q_c)  (pi: true,
q: predicted class rate) -- a class-specific re-scaling of the posteriors.  On
log-posteriors this is (to first order) an additive per-class offset, so we tune
b in  argmax_c (z_c + b_c),  b_0 = 0.

The objective is piecewise constant, so gradient-based or simplex methods stall;
each coordinate is maximised exactly over a grid with ``scipy.optimize.brute``
(coordinate ascent, a few sweeps), with a tiny ridge penalty that breaks ties
toward b = 0.  Callers must pass OOF scores whose labels exclude the fold being
scored (the ablation uses *inner* OOF of the training part: fully nested).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brute, minimize_scalar

try:
    from .losses import log_softmax, macro_f1
except ImportError:
    from losses import log_softmax, macro_f1


def logit_scale(Z):
    """Robust scale of class-relative logits; sets the bias search range."""
    centred = Z - Z.mean(axis=1, keepdims=True)
    return float(np.clip(np.std(centred), 0.05, 50.0))


def tune_class_bias(Z, y, n_rounds=4, grid_size=121, span=3.0, penalty=1e-3):
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    K = Z.shape[1]
    scale = logit_scale(Z)
    lo, hi = -span * scale, span * scale
    step = (hi - lo) / (grid_size - 1)
    b = np.zeros(K)

    def negative_score(value, cls):
        trial = b.copy()
        trial[cls] = float(np.ravel(value)[0])
        return -(macro_f1(y, (Z + trial).argmax(axis=1), K) - penalty * float(np.sum((trial / scale) ** 2)))

    for _ in range(n_rounds):
        previous = b.copy()
        for cls in range(1, K):
            best = brute(negative_score, ranges=(slice(lo, hi + step / 2, step),), args=(cls,), finish=None)
            b[cls] = float(np.ravel(best)[0])
        if np.allclose(previous, b):
            break
    return b


def apply_bias(Z, b):
    return (np.asarray(Z, dtype=np.float64) + np.asarray(b)[None, :]).argmax(axis=1)


def fit_temperature(Z, y, bounds=(0.05, 20.0)):
    """Temperature T minimising NLL(softmax(Z / T)); does not change argmax."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    idx = np.arange(len(y))

    def nll_at(log_t):
        return -float(log_softmax(Z / np.exp(log_t))[idx, y].mean())

    res = minimize_scalar(nll_at, bounds=tuple(np.log(bounds)), method='bounded')
    return float(np.exp(res.x))
