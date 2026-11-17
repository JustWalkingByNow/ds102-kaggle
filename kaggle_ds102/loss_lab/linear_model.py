"""From-scratch linear text classifier trained with any loss in ``losses.py``.

Model: z = X W + b over the repo's sparse TF-IDF/meta features (FeaturesV2),
W initialised at zero (default) or Glorot-uniform; no pretrained parameters.
Optimiser: full-batch L-BFGS on  L(z) + (1 / (2 C n)) ||W||_F^2  -- the same
"C per sample-sum" convention as sklearn's LinearSVC, so a C chosen on one
training size transfers to another exactly as in sklearn's refit.

Full-batch optimisation matters for the soft macro-F1 term: its value is a
ratio of sums over the batch, so mini-batch estimates are biased; here every
gradient is the exact gradient of the full-fold surrogate.

Composite losses are optimised by continuation: stage s minimises
alpha_s L_base + (1 - alpha_s) L_metric, warm-started from stage s-1, with
alpha_0 = 1 (pure base loss) and a cosine decay to alpha_min.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from losses import (CompositeLoss, FocalLoss, LogitAdjustedCrossEntropy, SoftMacroF1Loss,
                    SoftmaxCrossEntropy, SquaredHingeOvR, balanced_class_weights,
                    cosine_alpha_schedule, effective_number_weights, log_class_prior,
                    log_softmax, sqrt_balanced_class_weights)


def class_weights_from_spec(kind, y, K, beta=0.999):
    if kind in (None, 'none'):
        return None
    if kind == 'balanced':
        return balanced_class_weights(y, K)
    if kind == 'sqrt_balanced':
        return sqrt_balanced_class_weights(y, K)
    if kind == 'effective':
        return effective_number_weights(y, K, beta)
    raise ValueError(f'unknown class weight scheme {kind!r}')


def build_loss(spec, y, K):
    """Build a loss object from a JSON-serialisable spec using *training* labels only."""
    name = spec['name']
    cw = class_weights_from_spec(spec.get('weights'), y, K, spec.get('beta', 0.999))
    if name == 'ce':
        return SoftmaxCrossEntropy(class_weight=cw, label_smoothing=spec.get('label_smoothing', 0.0))
    if name == 'focal':
        return FocalLoss(gamma=spec.get('gamma', 2.0), class_weight=cw)
    if name == 'logit_adjusted':
        return LogitAdjustedCrossEntropy(log_class_prior(y, K), tau=spec.get('tau', 1.0), class_weight=cw)
    if name == 'squared_hinge_ovr':
        return SquaredHingeOvR(class_weight=cw)
    if name == 'soft_macro_f1':
        return SoftMacroF1Loss()
    if name == 'composite':
        return CompositeLoss(build_loss(spec['base'], y, K), build_loss(spec['metric'], y, K), alpha=1.0)
    raise ValueError(f'unknown loss {name!r}')


def stage_alphas(spec):
    if spec['name'] != 'composite':
        return [1.0]
    return cosine_alpha_schedule(int(spec.get('stages', 4)), float(spec.get('alpha_min', 0.25)))


class LinearLossClassifier:
    """sklearn-style estimator: fit / decision_function / predict_proba / predict."""

    def __init__(self, loss_spec, C=1.0, n_classes=4, max_iter=500, stage_max_iter=150,
                 tol=1e-9, init='zeros', random_state=102):
        self.loss_spec = loss_spec
        self.C = float(C)
        self.n_classes = int(n_classes)
        self.max_iter = int(max_iter)
        self.stage_max_iter = int(stage_max_iter)
        self.tol = float(tol)
        self.init = init
        self.random_state = random_state

    # -- internals -------------------------------------------------------------
    def _initial_theta(self, d, K):
        if self.init == 'zeros':
            return np.zeros(d * K + K)
        if self.init == 'xavier':
            rng = np.random.default_rng(self.random_state)
            limit = np.sqrt(6.0 / (d + K))
            return np.concatenate([rng.uniform(-limit, limit, d * K), np.zeros(K)])
        raise ValueError(f'unknown init {self.init!r}')

    def _objective(self, theta, X, XT, y, loss, l2, sample_weight):
        d, K = X.shape[1], self.n_classes
        W = theta[:d * K].reshape(d, K)
        b = theta[d * K:]
        Z = X @ W + b
        value, dZ = loss(Z, y, sample_weight)
        dW = XT @ dZ + l2 * W
        total = value + 0.5 * l2 * float(np.vdot(W, W))
        grad = np.concatenate([dW.ravel(), dZ.sum(axis=0)])
        if not (np.isfinite(total) and np.isfinite(grad).all()):
            raise FloatingPointError('non-finite objective or gradient')
        self._last_data_loss = value
        return total, grad

    # -- public API ------------------------------------------------------------
    def fit(self, X, y, sample_weight=None, theta_init=None):
        X = sparse.csr_matrix(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        K = self.n_classes
        if set(np.unique(y)) != set(range(K)):
            raise ValueError('training fold must contain every class')
        XT = X.T.tocsr()
        loss = build_loss(self.loss_spec, y, K)
        l2 = 1.0 / (self.C * n)
        theta = self._initial_theta(d, K) if theta_init is None else np.array(theta_init, dtype=np.float64)
        self.stages_ = []
        start = time.time()
        for s, alpha in enumerate(stage_alphas(self.loss_spec)):
            if isinstance(loss, CompositeLoss):
                loss.alpha = alpha
            res = minimize(self._objective, theta, args=(X, XT, y, loss, l2, sample_weight),
                           jac=True, method='L-BFGS-B',
                           options=dict(maxiter=self.max_iter if s == 0 else self.stage_max_iter,
                                        maxcor=20, ftol=self.tol, gtol=1e-8))
            theta = res.x
            self.stages_.append(dict(alpha=float(alpha), n_iter=int(res.nit), objective=float(res.fun),
                                     data_loss=float(self._last_data_loss),
                                     converged=bool(res.success), message=str(res.message)))
        self.fit_seconds_ = time.time() - start
        self.coef_ = theta[:d * K].reshape(d, K)
        self.intercept_ = theta[d * K:]
        self.theta_ = theta
        self.loss_ = loss
        # final training loss (the loss actually optimised in the last stage, no L2 term)
        self.train_loss_ = float(loss(self.decision_function(X), y, sample_weight)[0])
        return self

    def decision_function(self, X):
        return np.asarray(sparse.csr_matrix(X, dtype=np.float64) @ self.coef_) + self.intercept_

    def predict_proba(self, X):
        return np.exp(log_softmax(self.decision_function(X)))

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)

    def loss_on(self, X, y):
        """Value of the trained loss (training-fold statistics) on another split."""
        return float(self.loss_(self.decision_function(X), np.asarray(y, dtype=np.int64))[0])
