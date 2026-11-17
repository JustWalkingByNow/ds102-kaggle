"""Numerically stable loss functions for DS102 (4-class, macro-F1), from scratch.

Every loss works on raw logits ``Z`` (shape ``n x K``) and integer labels ``y``.
``loss(Z, y, sample_weight)`` returns ``(value, dZ)`` where ``dZ`` is the exact
analytic gradient of the *weighted mean* loss with respect to ``Z``.  Models
(linear, MLP, ...) only need ``dZ`` and the chain rule; nothing here touches
pretrained weights or external data.

Numerical-stability rules applied throughout
--------------------------------------------
* probabilities come from ``log_softmax`` (log-sum-exp with max shift), never
  from ``exp`` of raw logits;
* ``1 - p`` is computed as ``-expm1(log p)`` and clamped at ``EPS = 1e-7``
  before any negative power;
* every denominator (soft-F1 counts, class counts) is bounded away from zero.

Fold safety: class weights, priors and effective numbers are *functions of the
training labels passed in*.  The trainer builds the loss from ``y_train`` of the
current fold only (see ``linear_model.LinearLossClassifier``), so no validation
statistic can leak into a loss.

Losses (equations use p = softmax(z), t = true class, pi = class prior)
----------------------------------------------------------------------
* ``SoftmaxCrossEntropy``   l = -sum_c q_c log p_c,  q = (1-eps) onehot(t) + eps/K
* ``FocalLoss``             l = -a_t (1-p_t)^g log p_t                (Lin et al. 2017)
* ``LogitAdjustedCrossEntropy``  l = -log softmax(z + tau log pi)_t   (Menon et al. 2021)
* ``SoftMacroF1Loss``       L = 1 - (1/K) sum_c 2 TP_c / (sum_i p_ic + N_c)
* ``SquaredHingeOvR``       l = sum_c W_ic max(0, 1 - T_ic z_ic)^2   (liblinear L2-SVM)
* ``CompositeLoss``         L = a L_base + (1 - a) L_metric,  a annealed by stage
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

EPS = 1e-7


# --------------------------------------------------------------------------- helpers
def log_softmax(Z):
    """Row-wise log-softmax via log-sum-exp (max-shifted inside scipy)."""
    Z = np.asarray(Z, dtype=np.float64)
    return Z - logsumexp(Z, axis=1, keepdims=True)


def softmax(Z):
    return np.exp(log_softmax(Z))


def _validate(Z, y, sample_weight):
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    if Z.ndim != 2 or y.ndim != 1 or len(y) != Z.shape[0]:
        raise ValueError(f'shape mismatch: Z{Z.shape} y{y.shape}')
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= Z.shape[1]:
        raise ValueError('y must hold integer labels in [0, K)')
    if not np.isfinite(Z).all():
        raise FloatingPointError('non-finite logits')
    if sample_weight is None:
        s = np.ones(len(y))
    else:
        s = np.asarray(sample_weight, dtype=np.float64)
        if s.shape != y.shape or (s < 0).any() or not np.isfinite(s).all() or s.sum() <= 0:
            raise ValueError('sample_weight must be finite, non-negative, not all zero')
    return Z, y.astype(np.int64), s


def _onehot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def class_counts(y, K):
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=K).astype(np.float64)


def balanced_class_weights(y, K):
    """sklearn 'balanced': w_c = n / (K n_c).  Satisfies sum_i w_{y_i} = n."""
    n_c = class_counts(y, K)
    if (n_c == 0).any():
        raise ValueError('every class must appear in the training fold')
    return len(y) / (K * n_c)


def sqrt_balanced_class_weights(y, K):
    """The repo's 'sqrt_balanced' (train_submit.make_model), renormalised to mean 1 per sample."""
    w = np.sqrt(balanced_class_weights(y, K))
    return w * len(y) / float((w * class_counts(y, K)).sum())


def effective_number_weights(y, K, beta=0.999):
    """Class-Balanced weights (Cui et al. 2019): a_c ~ (1 - beta) / (1 - beta^n_c).

    Normalised so that sum_i a_{y_i} = n (mean per-sample weight 1), which keeps
    the data term on the same scale as the unweighted loss for a fixed L2.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError('beta must lie in [0, 1)')
    n_c = class_counts(y, K)
    if (n_c == 0).any():
        raise ValueError('every class must appear in the training fold')
    effective = -np.expm1(n_c * np.log(beta)) / (1.0 - beta) if beta > 0 else np.ones(K)
    w = 1.0 / effective
    return w * len(y) / float((w * n_c).sum())


def log_class_prior(y, K):
    n_c = class_counts(y, K)
    return np.log(np.maximum(n_c, 1.0) / float(len(y)))


# --------------------------------------------------------------------------- losses
class SoftmaxCrossEntropy:
    """Weighted softmax cross-entropy with optional label smoothing.

    l_i = -sum_c q_ic log p_ic,   dl_i/dz_i = p_i - q_i.
    """

    def __init__(self, class_weight=None, label_smoothing=0.0):
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError('label_smoothing must lie in [0, 1)')
        self.class_weight = None if class_weight is None else np.asarray(class_weight, dtype=np.float64)
        self.label_smoothing = float(label_smoothing)

    def __call__(self, Z, y, sample_weight=None):
        Z, y, s = _validate(Z, y, sample_weight)
        n, K = Z.shape
        if self.class_weight is not None:
            s = s * self.class_weight[y]
        logp = log_softmax(Z)
        q = _onehot(y, K) * (1.0 - self.label_smoothing) + self.label_smoothing / K
        per_sample = -(q * logp).sum(axis=1)
        denom = s.sum()
        value = float((s * per_sample).sum() / denom)
        dZ = (np.exp(logp) - q) * (s / denom)[:, None]
        return value, dZ


class FocalLoss:
    """Multi-class softmax focal loss, optionally class-balanced.

    l_i = -a_t (1 - p_t)^g log p_t.  With u = p_t and
    G(u) = (1-u)^g - g u (1-u)^(g-1) log u  (>= 0),  the gradient is
    dl_i/dz_ij = a_t G(u) (p_ij - [j == t]).   g = 0 recovers cross-entropy.
    """

    def __init__(self, gamma=2.0, class_weight=None):
        if gamma < 0:
            raise ValueError('gamma must be non-negative')
        self.gamma = float(gamma)
        self.class_weight = None if class_weight is None else np.asarray(class_weight, dtype=np.float64)

    def modulating_terms(self, logp_t):
        """Return (f(u), G(u)) where f(u) = -(1-u)^g log u, evaluated stably."""
        g = self.gamma
        u = np.exp(logp_t)
        one_minus = np.maximum(-np.expm1(logp_t), EPS)          # 1 - u, never 0
        pow_g = one_minus ** g
        f = -pow_g * logp_t
        if g == 0.0:
            G = np.ones_like(u)
        else:
            G = pow_g - g * u * one_minus ** (g - 1.0) * logp_t
        return f, G

    def __call__(self, Z, y, sample_weight=None):
        Z, y, s = _validate(Z, y, sample_weight)
        n, K = Z.shape
        if self.class_weight is not None:
            s = s * self.class_weight[y]
        logp = log_softmax(Z)
        logp_t = logp[np.arange(n), y]
        f, G = self.modulating_terms(logp_t)
        denom = s.sum()
        value = float((s * f).sum() / denom)
        dZ = np.exp(logp) - _onehot(y, K)
        dZ *= (s * G / denom)[:, None]
        return value, dZ


class LogitAdjustedCrossEntropy:
    """Logit-adjusted softmax CE (Menon et al., ICLR 2021).

    Training uses z + tau * log(pi); prediction uses the raw z, so the model's
    own argmax already applies the prior correction.  In the near-separable
    TF-IDF regime this enforces class-dependent margins (larger for rare
    classes), which plain re-weighting cannot (Byrd & Lipton 2019).
    """

    def __init__(self, log_prior, tau=1.0, class_weight=None):
        self.offset = float(tau) * np.asarray(log_prior, dtype=np.float64)
        self.ce = SoftmaxCrossEntropy(class_weight=class_weight)

    def __call__(self, Z, y, sample_weight=None):
        Z = np.asarray(Z, dtype=np.float64)
        return self.ce(Z + self.offset[None, :], y, sample_weight)


class SoftMacroF1Loss:
    """Differentiable surrogate of 1 - macro-F1 over the whole batch.

    TP_c = sum_i s_i y_ic p_ic,  D_c = sum_i s_i p_ic + sum_i s_i y_ic (+EPS),
    F1_c = 2 TP_c / D_c,  L = 1 - mean_c F1_c.
    dF1_c/dp_ic = (2 s_i / D_c) (y_ic - F1_c / 2): each class is automatically
    re-weighted by 1/D_c and every sample is pulled towards the F1-optimal
    threshold F1_c / 2 -- the same structure as the Bayes-optimal macro-F1 rule.
    Non-decomposable: use it full-batch (as the L-BFGS trainer does) or with
    large batches.
    """

    def __init__(self, eps=EPS):
        self.eps = float(eps)

    def per_class_f1(self, Z, y, sample_weight=None):
        Z, y, s = _validate(Z, y, sample_weight)
        P = np.exp(log_softmax(Z))
        Y = _onehot(y, Z.shape[1])
        tp = (s[:, None] * Y * P).sum(axis=0)
        D = (s[:, None] * P).sum(axis=0) + (s[:, None] * Y).sum(axis=0) + self.eps
        return 2.0 * tp / D

    def __call__(self, Z, y, sample_weight=None):
        Z, y, s = _validate(Z, y, sample_weight)
        n, K = Z.shape
        P = np.exp(log_softmax(Z))
        Y = _onehot(y, K)
        tp = (s[:, None] * Y * P).sum(axis=0)
        D = (s[:, None] * P).sum(axis=0) + (s[:, None] * Y).sum(axis=0) + self.eps
        f1 = 2.0 * tp / D
        value = float(1.0 - f1.mean())
        dP = -(2.0 / K) * s[:, None] * (Y - 0.5 * f1[None, :]) / D[None, :]
        # back-propagate through softmax: dZ_ij = p_ij (dP_ij - sum_c dP_ic p_ic)
        dZ = P * (dP - (dP * P).sum(axis=1, keepdims=True))
        return value, dZ


class SquaredHingeOvR:
    """One-vs-rest squared hinge, i.e. the loss inside sklearn's LinearSVC.

    T_ic = +1 if y_i == c else -1.  liblinear weights the *positive* side of
    sub-problem c by class_weight[c] and the negative side by 1, reproduced
    here as W_ic = w_c if y_i == c else 1.
    l_i = sum_c W_ic max(0, 1 - T_ic z_ic)^2.
    """

    def __init__(self, class_weight=None):
        self.class_weight = None if class_weight is None else np.asarray(class_weight, dtype=np.float64)

    def __call__(self, Z, y, sample_weight=None):
        Z, y, s = _validate(Z, y, sample_weight)
        n, K = Z.shape
        Y = _onehot(y, K)
        T = 2.0 * Y - 1.0
        W = np.ones((n, K))
        if self.class_weight is not None:
            W = np.where(Y > 0, self.class_weight[None, :], 1.0)
        slack = np.maximum(0.0, 1.0 - T * Z)
        denom = s.sum()
        value = float((s[:, None] * W * slack ** 2).sum() / denom)
        dZ = -2.0 * s[:, None] * W * T * slack / denom
        return value, dZ


class CompositeLoss:
    """L = alpha * L_base + (1 - alpha) * L_metric; ``alpha`` is set per stage."""

    def __init__(self, base, metric, alpha=1.0):
        self.base, self.metric = base, metric
        self.alpha = float(alpha)

    def __call__(self, Z, y, sample_weight=None):
        a = self.alpha
        vb, gb = self.base(Z, y, sample_weight)
        if a >= 1.0:
            return vb, gb
        vm, gm = self.metric(Z, y, sample_weight)
        return a * vb + (1.0 - a) * vm, a * gb + (1.0 - a) * gm


def cosine_alpha_schedule(n_stages, alpha_min):
    """alpha_s = alpha_min + (1 - alpha_min) * (1 + cos(pi s / (S - 1))) / 2, s = 0..S-1.

    Stage 0 is pure base loss (alpha = 1): the from-scratch model first learns a
    stable, calibrated solution before the metric surrogate sharpens it.
    """
    if n_stages < 1 or not 0.0 <= alpha_min <= 1.0:
        raise ValueError('need n_stages >= 1 and alpha_min in [0, 1]')
    if n_stages == 1:
        return [1.0]
    s = np.arange(n_stages)
    return (alpha_min + (1.0 - alpha_min) * 0.5 * (1.0 + np.cos(np.pi * s / (n_stages - 1)))).tolist()


# --------------------------------------------------------------------------- metrics
def macro_f1(y, pred, K=4):
    """Exact competition metric (F1-macro over labels 0..K-1, zero_division=0)."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(K * y + pred, minlength=K * K).reshape(K, K)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2 * np.diag(cm), den, out=np.zeros(K), where=den > 0)))


def per_class_f1(y, pred, K=4):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(K * y + pred, minlength=K * K).reshape(K, K)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2 * np.diag(cm), den, out=np.zeros(K), where=den > 0)


def nll(Z, y):
    """Mean negative log-likelihood of softmax(Z) (common calibration yardstick)."""
    logp = log_softmax(Z)
    return float(-logp[np.arange(len(y)), np.asarray(y)].mean())


# --------------------------------------------------------------------------- GBDT objectives
def focal_grad_hess(raw, y, gamma=2.0, class_weight=None, hess_floor=1e-6, hess_scale=1.0):
    """Exact gradient and exact diagonal Hessian of softmax focal loss per sample.

    With u = p_t, d_j = [j == t] - p_j and G as in ``FocalLoss``:
      g_ij = a_t G(u) (p_ij - [j == t])
      H_ij = a_t [ -G'(u) u d_j^2 + G(u) p_j (1 - p_j) ]
      G'(u) = -2g(1-u)^(g-1) - g(1-u)^(g-1) log u + g(g-1) u (1-u)^(g-2) log u
    The exact diagonal can be negative for badly misclassified rows (focal loss
    is not convex in the logits), so the returned Hessian is floored at
    ``hess_floor * a_t`` to keep LightGBM's Newton leaf values well defined.
    Returns grad, hess_used, hess_exact (all n x K, *sum*-scaled per sample).
    """
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, K = raw.shape
    a = np.ones(n) if class_weight is None else np.asarray(class_weight, dtype=np.float64)[y]
    logp = log_softmax(raw)
    P = np.exp(logp)
    logu = logp[np.arange(n), y]
    u = np.exp(logu)
    one_minus = np.maximum(-np.expm1(logu), EPS)
    gm = float(gamma)
    if gm == 0.0:
        G = np.ones(n)
        dG = np.zeros(n)
    else:
        G = one_minus ** gm - gm * u * one_minus ** (gm - 1.0) * logu
        dG = (-2.0 * gm * one_minus ** (gm - 1.0) - gm * one_minus ** (gm - 1.0) * logu
              + gm * (gm - 1.0) * u * one_minus ** (gm - 2.0) * logu)
    Y = _onehot(y, K)
    d = Y - P
    grad = a[:, None] * G[:, None] * (P - Y)
    hess_exact = a[:, None] * (-dG[:, None] * u[:, None] * d ** 2 + G[:, None] * P * (1.0 - P))
    hess = np.maximum(hess_scale * hess_exact, hess_floor * a[:, None])
    return grad, hess, hess_exact


def make_lgb_focal_objective(gamma=2.0, class_weight=None, hess_floor=1e-6, hess_scale=1.0):
    """LightGBM >= 4 custom multiclass objective: f(preds, data) -> (grad, hess), n x K.

    ``preds`` already include the Dataset init_score.  Per-row weights set on the
    Dataset are applied here (LightGBM does not re-weight custom gradients).
    """
    def objective(preds, train_data):
        y = np.asarray(train_data.get_label(), dtype=np.int64)
        raw = np.asarray(preds, dtype=np.float64).reshape(len(y), -1)
        grad, hess, _ = focal_grad_hess(raw, y, gamma, class_weight, hess_floor, hess_scale)
        w = train_data.get_weight()
        if w is not None:
            w = np.asarray(w, dtype=np.float64)[:, None]
            grad, hess = grad * w, hess * w
        if not (np.isfinite(grad).all() and np.isfinite(hess).all() and (hess > 0).all()):
            raise FloatingPointError('invalid focal gradient/Hessian')
        return grad, hess
    return objective


def lgb_macro_f1_eval(preds, eval_data):
    """LightGBM feval mirroring the competition metric (argmax of raw scores)."""
    y = np.asarray(eval_data.get_label(), dtype=np.int64)
    raw = np.asarray(preds, dtype=np.float64).reshape(len(y), -1)
    return 'macro_f1', macro_f1(y, raw.argmax(axis=1), raw.shape[1]), True
