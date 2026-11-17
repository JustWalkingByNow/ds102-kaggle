"""Unit checks for loss_lab: values, analytic gradients, Hessians, stability.

Run from kaggle_ds102/:  python -m unittest loss_lab.test_losses -v
(or from loss_lab/:      python -m unittest test_losses -v)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from losses import (CompositeLoss, FocalLoss, LogitAdjustedCrossEntropy, SoftMacroF1Loss,  # noqa: E402
                    SoftmaxCrossEntropy, SquaredHingeOvR, balanced_class_weights,
                    cosine_alpha_schedule, effective_number_weights, focal_grad_hess,
                    lgb_macro_f1_eval, log_class_prior, macro_f1, make_lgb_focal_objective)
from linear_model import LinearLossClassifier, build_loss  # noqa: E402
from postprocess import apply_bias, fit_temperature, tune_class_bias  # noqa: E402

K = 4


def numeric_grad(fun, Z, h=1e-6):
    G = np.zeros_like(Z)
    for idx in np.ndindex(*Z.shape):
        up, dn = Z.copy(), Z.copy()
        up[idx] += h
        dn[idx] -= h
        G[idx] = (fun(up) - fun(dn)) / (2 * h)
    return G


def all_losses(y):
    w = balanced_class_weights(y, K)
    cb = effective_number_weights(y, K, 0.999)
    return {
        'ce': SoftmaxCrossEntropy(),
        'ce_balanced': SoftmaxCrossEntropy(class_weight=w),
        'ce_label_smoothing': SoftmaxCrossEntropy(label_smoothing=0.1),
        'focal_g0': FocalLoss(gamma=0.0),
        'focal_g0.5': FocalLoss(gamma=0.5),
        'focal_g2_cb': FocalLoss(gamma=2.0, class_weight=cb),
        'logit_adjusted': LogitAdjustedCrossEntropy(log_class_prior(y, K), tau=1.0),
        'soft_macro_f1': SoftMacroF1Loss(),
        'squared_hinge_ovr': SquaredHingeOvR(class_weight=w),
        'composite_half': CompositeLoss(SoftmaxCrossEntropy(class_weight=w), SoftMacroF1Loss(), alpha=0.5),
    }


class GradientChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.array([0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 0, 1, 3])
        self.Z = rng.normal(scale=2.0, size=(len(self.y), K))
        self.s = rng.uniform(0.5, 2.0, size=len(self.y))

    def test_analytic_gradients_match_finite_differences(self):
        for name, loss in all_losses(self.y).items():
            for weights in (None, self.s):
                with self.subTest(loss=name, weighted=weights is not None):
                    value, grad = loss(self.Z, self.y, weights)
                    num = numeric_grad(lambda Z: loss(Z, self.y, weights)[0], self.Z)
                    self.assertTrue(np.isfinite(value))
                    err = np.abs(grad - num).max() / max(np.abs(num).max(), 1e-12)
                    self.assertLess(err, 1e-6, f'{name}: relative gradient error {err:.2e}')
                    self.assertGreater(np.var(grad), 0.0, f'{name}: degenerate (constant) gradient')

    def test_focal_gamma0_equals_cross_entropy(self):
        w = balanced_class_weights(self.y, K)
        v1, g1 = FocalLoss(0.0, w)(self.Z, self.y)
        v2, g2 = SoftmaxCrossEntropy(w)(self.Z, self.y)
        self.assertAlmostEqual(v1, v2, places=12)
        np.testing.assert_allclose(g1, g2, atol=1e-14)

    def test_logit_adjustment_tau0_equals_cross_entropy(self):
        v1, g1 = LogitAdjustedCrossEntropy(log_class_prior(self.y, K), tau=0.0)(self.Z, self.y)
        v2, g2 = SoftmaxCrossEntropy()(self.Z, self.y)
        self.assertAlmostEqual(v1, v2, places=12)
        np.testing.assert_allclose(g1, g2, atol=1e-14)

    def test_composite_endpoints(self):
        base, metric = SoftmaxCrossEntropy(), SoftMacroF1Loss()
        for alpha, ref in ((1.0, base), (0.0, metric)):
            v, g = CompositeLoss(base, metric, alpha)(self.Z, self.y)
            vr, gr = ref(self.Z, self.y)
            self.assertAlmostEqual(v, vr, places=12)
            np.testing.assert_allclose(g, gr, atol=1e-14)

    def test_softmax_gradients_sum_to_zero_per_row(self):
        # shift-invariance of softmax losses: sum_j dL/dz_ij = 0
        for name in ['ce', 'ce_balanced', 'focal_g2_cb', 'logit_adjusted', 'soft_macro_f1', 'composite_half']:
            _, g = all_losses(self.y)[name](self.Z, self.y)
            np.testing.assert_allclose(g.sum(axis=1), 0.0, atol=1e-14, err_msg=name)


class StabilityChecks(unittest.TestCase):
    def test_extreme_logits_stay_finite(self):
        y = np.array([0, 1, 2, 3, 0, 1])
        for scale in (1e2, 7e2, 1e4, 1e8):
            Z = np.array([[scale, -scale, 0, 0], [-scale, scale, 0, 0], [scale, 0, -scale, 0],
                          [0, 0, 0, scale], [-scale, 0, 0, scale], [0, -scale, scale, 0]], dtype=float)
            for name, loss in all_losses(y).items():
                with self.subTest(loss=name, scale=scale):
                    v, g = loss(Z, y)
                    self.assertTrue(np.isfinite(v), name)
                    self.assertTrue(np.isfinite(g).all(), name)
            grad, hess, exact = focal_grad_hess(Z, y, gamma=2.0)
            self.assertTrue(np.isfinite(grad).all() and np.isfinite(exact).all() and (hess > 0).all())

    def test_non_finite_logits_are_rejected(self):
        with self.assertRaises(FloatingPointError):
            SoftmaxCrossEntropy()(np.array([[np.nan, 0, 0, 0]]), np.array([0]))

    def test_soft_f1_saturates_to_hard_f1(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, K, 200)
        pred = np.where(rng.random(200) < 0.7, y, rng.integers(0, K, 200))
        Z = np.full((200, K), -50.0)
        Z[np.arange(200), pred] = 50.0
        soft = 1.0 - SoftMacroF1Loss()(Z, y)[0]
        self.assertAlmostEqual(soft, macro_f1(y, pred), places=6)


class WeightsAndMetric(unittest.TestCase):
    def test_class_weights_are_fold_statistics(self):
        y = np.array([0] * 80 + [1] * 2 + [2] * 8 + [3] * 10)
        w = balanced_class_weights(y, K)
        self.assertAlmostEqual(float(w[y].sum()), len(y))
        cb = effective_number_weights(y, K, 0.999)
        self.assertAlmostEqual(float(cb[y].sum()), len(y))
        self.assertTrue(np.all(np.diff(cb[[0, 3, 2, 1]]) > 0))     # rarer -> heavier
        np.testing.assert_allclose(effective_number_weights(y, K, 0.0), 1.0)
        # beta -> 1 approaches inverse frequency
        near = effective_number_weights(y, K, 1 - 1e-9)
        np.testing.assert_allclose(near / near[0], w / w[0], rtol=1e-5)
        # a different training fold gives different statistics (nothing global)
        y2 = np.array([0] * 50 + [1] * 20 + [2] * 20 + [3] * 10)
        self.assertFalse(np.allclose(balanced_class_weights(y2, K), w))

    def test_macro_f1_matches_sklearn(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(2)
        for _ in range(20):
            y = rng.integers(0, K, 300)
            p = rng.integers(0, K, 300)
            p[p == 1] = 0                     # include a never-predicted class
            ref = f1_score(y, p, labels=list(range(K)), average='macro', zero_division=0)
            self.assertAlmostEqual(macro_f1(y, p), ref, places=12)

    def test_schedule(self):
        a = cosine_alpha_schedule(4, 0.25)
        self.assertEqual(a[0], 1.0)
        self.assertAlmostEqual(a[-1], 0.25)
        self.assertTrue(all(x >= y for x, y in zip(a, a[1:])))


class GBDTObjective(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = rng.integers(0, K, 9)
        self.raw = rng.normal(scale=3.0, size=(9, K))
        self.cw = np.array([1.0, 5.0, 2.0, 1.5])

    def test_focal_grad_and_exact_hessian(self):
        for gamma in (0.0, 0.5, 1.0, 2.0, 3.0):
            grad, hess, exact = focal_grad_hess(self.raw, self.y, gamma, self.cw)
            fl = FocalLoss(gamma, self.cw)

            def per_row_sum(R):  # focal loss summed over rows (LightGBM convention)
                a = self.cw[self.y]
                v, _ = fl(R, self.y)
                return v * a.sum()

            num = numeric_grad(per_row_sum, self.raw)
            np.testing.assert_allclose(grad, num, rtol=1e-6, atol=1e-8, err_msg=f'grad gamma={gamma}')
            # exact diagonal Hessian vs finite differences of the analytic gradient
            h = 1e-5
            num_h = np.zeros_like(self.raw)
            for i in range(self.raw.shape[0]):
                for j in range(K):
                    up, dn = self.raw.copy(), self.raw.copy()
                    up[i, j] += h
                    dn[i, j] -= h
                    num_h[i, j] = (focal_grad_hess(up, self.y, gamma, self.cw)[0][i, j]
                                   - focal_grad_hess(dn, self.y, gamma, self.cw)[0][i, j]) / (2 * h)
            np.testing.assert_allclose(exact, num_h, rtol=1e-5, atol=1e-7, err_msg=f'hess gamma={gamma}')
            self.assertTrue((hess > 0).all(), f'non-positive Hessian used, gamma={gamma}')

    def test_gamma0_hessian_is_softmax_ce_hessian(self):
        from losses import softmax
        grad, hess, exact = focal_grad_hess(self.raw, self.y, 0.0)
        P = softmax(self.raw)
        np.testing.assert_allclose(exact, P * (1 - P), atol=1e-14)

    def test_lightgbm_custom_objective_runs(self):
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest('lightgbm not installed')
        rng = np.random.default_rng(4)
        X = rng.normal(size=(600, 5))
        y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0.8)
        y[y > 3] = 3
        ds = lgb.Dataset(X, y, free_raw_data=False)
        params = dict(objective=make_lgb_focal_objective(2.0), num_class=K, learning_rate=0.1,
                      num_leaves=7, verbose=-1, seed=0, deterministic=True, num_threads=1)
        booster = lgb.train(params, ds, num_boost_round=30, valid_sets=[ds], feval=lgb_macro_f1_eval,
                            callbacks=[lgb.record_evaluation(rec := {})])
        raw = booster.predict(X, raw_score=True)
        self.assertEqual(raw.shape, (600, K))
        self.assertTrue(np.isfinite(raw).all())
        self.assertGreater(rec['training']['macro_f1'][-1], 0.6)

    def test_custom_ce_reproduces_builtin_multiclass(self):
        """gamma=0, Hessian x K/(K-1), log-prior init, weighted rows == LightGBM 'multiclass'."""
        try:
            import lightgbm as lgb
        except ImportError:
            self.skipTest('lightgbm not installed')
        from losses import softmax
        rng = np.random.default_rng(7)
        X = rng.normal(size=(1500, 5))
        y = np.minimum((X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 1.2) + (rng.random(1500) < .05), 3)
        w = np.sqrt(len(y) / (4 * np.bincount(y, minlength=4)))[y]
        init = np.tile(np.log(np.bincount(y, weights=w, minlength=4) / w.sum()), (len(y), 1))
        base = dict(num_class=K, learning_rate=0.1, num_leaves=7, verbose=-1, seed=0, deterministic=True, num_threads=1)
        custom = lgb.train(dict(base, objective=make_lgb_focal_objective(0.0, hess_scale=K / (K - 1))),
                           lgb.Dataset(X, y, weight=w, init_score=init), num_boost_round=40)
        builtin = lgb.train(dict(base, objective='multiclass'), lgb.Dataset(X, y, weight=w), num_boost_round=40)
        p1 = softmax(custom.predict(X, raw_score=True) + init)
        p2 = softmax(builtin.predict(X, raw_score=True))
        self.assertLess(np.abs(p1 - p2).max(), 1e-6)


class TrainerAndPostprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        n, d = 900, 30
        cls.X = rng.normal(size=(n, d))
        logits = cls.X[:, :K] * 2.0 + np.array([2.5, -2.5, -0.5, 0.5])
        cls.y = np.array([rng.choice(K, p=np.exp(r - r.max()) / np.exp(r - r.max()).sum()) for r in logits])

    def test_trainer_converges_for_every_recipe(self):
        specs = [dict(name='ce', weights='balanced'), dict(name='focal', gamma=2.0, weights='effective'),
                 dict(name='logit_adjusted', tau=1.0), dict(name='squared_hinge_ovr', weights='balanced'),
                 dict(name='composite', base=dict(name='ce', weights='balanced'),
                      metric=dict(name='soft_macro_f1'), stages=3, alpha_min=0.25)]
        for spec in specs:
            with self.subTest(spec=spec['name']):
                m = LinearLossClassifier(spec, C=1.0, max_iter=300).fit(self.X, self.y)
                self.assertTrue(all(np.isfinite(s['objective']) for s in m.stages_))
                self.assertTrue(m.stages_[0]['converged'], m.stages_[0]['message'])
                self.assertGreater(macro_f1(self.y, m.predict(self.X)), 0.45)

    def test_squared_hinge_replica_matches_liblinear(self):
        from sklearn.svm import LinearSVC
        y, X = self.y, self.X
        C = 0.5
        ref = LinearSVC(C=C, class_weight='balanced', dual=False, tol=1e-10, max_iter=100000,
                        intercept_scaling=1e3).fit(X, y)
        mine = LinearLossClassifier(dict(name='squared_hinge_ovr', weights='balanced'),
                                    C=C, max_iter=5000, tol=1e-14).fit(X, y)
        a, b = ref.decision_function(X), mine.decision_function(X)
        self.assertLess(np.abs(a - b).max(), 2e-3)
        self.assertGreater((a.argmax(1) == b.argmax(1)).mean(), 0.995)

    def test_build_loss_uses_only_given_labels(self):
        loss = build_loss(dict(name='ce', weights='balanced'), self.y[:300], K)
        np.testing.assert_allclose(loss.class_weight, balanced_class_weights(self.y[:300], K))

    def test_bias_tuning_never_hurts_in_sample_and_keeps_b0(self):
        m = LinearLossClassifier(dict(name='ce'), C=1.0).fit(self.X, self.y)
        Z = m.decision_function(self.X)
        b = tune_class_bias(Z, self.y)
        self.assertEqual(b[0], 0.0)
        self.assertGreaterEqual(macro_f1(self.y, apply_bias(Z, b)) + 1e-12,
                                macro_f1(self.y, Z.argmax(1)) - 1e-3)

    def test_temperature_recovers_known_scale(self):
        rng = np.random.default_rng(6)
        Z = rng.normal(size=(5000, K)) * 1.5
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        y = np.array([rng.choice(K, p=p) for p in P])
        self.assertAlmostEqual(fit_temperature(3.0 * Z, y), 3.0, delta=0.25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
