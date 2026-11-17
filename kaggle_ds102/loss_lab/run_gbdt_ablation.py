"""Nested CV ablation of LightGBM objectives (family B) on the repo's latent features.

Features: ``latent_v6.LatentFeatures`` with 128 SVD components (the V11 setting),
refitted from scratch on every training part.  Booster: the repo's LightGBM
recipe (lightgbm_latent_v18.PARAMS) with the native API.  Only the objective
changes between rows:

  G0  built-in 'multiclass', SqrtBalanced row weights            (repo v18 recipe)
  G1  custom softmax CE (focal g=0), same weights, Hessian x K/(K-1).  Bit-identical to G0
      when feature_fraction = 1 (unit-tested); with the repo's feature_fraction = 0.85 the
      column-sampling RNG stream differs, so G1 - G0 measures LightGBM's seed noise floor
  G2  custom class-balanced focal (g=2, beta=0.999), exact Hessian floored > 0
  G3  logit-adjusted CE: init_score = tau log(pi) in training only

Protocol (identical outer/inner grouped folds to run_ablation.py, seed 102):
  inner 3-fold on the outer-train part: early stopping with the macro-F1 feval on
  the *inner* validation fold only (patience 100, <= 1500 rounds) -> inner OOF;
  class offsets + temperature fitted on inner OOF; outer refit with the mean
  inner best iteration; outer fold scored raw and with offsets.

    python loss_lab/run_gbdt_ablation.py --data-dir <csv dir> --out loss_lab/output/gbdt_seed102
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_ablation as ra  # noqa: E402
from data import SEED, grouped_splits, guard_not_macos, load, sha256  # noqa: E402
from losses import (FocalLoss, LogitAdjustedCrossEntropy, SoftmaxCrossEntropy,  # noqa: E402
                    effective_number_weights, lgb_macro_f1_eval, log_class_prior,
                    macro_f1, make_lgb_focal_objective, nll, per_class_f1, sqrt_balanced_class_weights)
from postprocess import apply_bias, fit_temperature, tune_class_bias  # noqa: E402

K = 4
PARAMS = dict(num_class=K, learning_rate=0.05, num_leaves=31, max_depth=-1, min_data_in_leaf=25,
              min_gain_to_split=0.02, lambda_l1=0.1, lambda_l2=10.0, bagging_fraction=0.85, bagging_freq=1,
              feature_fraction=0.85, seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED,
              data_random_seed=SEED, deterministic=True, force_col_wise=True, verbosity=-1,
              num_threads=1, metric='None')
MAX_ROUNDS, PATIENCE = 1500, 100
CONFIGS = {
    'G0_builtin_multiclass_sqrtbal': dict(objective='builtin', weights='sqrt_balanced',
                                          role='repo v18 recipe: built-in multiclass + SqrtBalanced'),
    'G1_custom_ce_sqrtbal': dict(objective='focal', gamma=0.0, weights='sqrt_balanced',
                                 role='same loss as G0 via custom objective; differs only by LightGBM column-sampling RNG -> noise floor'),
    'G2_custom_cb_focal': dict(objective='focal', gamma=2.0, weights='effective', beta=0.999,
                               role='candidate: class-balanced focal, exact Hessian (floored)'),
    'G3_logit_adjusted': dict(objective='builtin', weights=None, logit_adjust_tau=1.0,
                              role='candidate: logit-adjusted CE via training init_score'),
    # robustness baselines added after G3's early-stopping depth was observed (seed 102):
    'G0LL_builtin_sqrtbal_loglossES': dict(objective='builtin', weights='sqrt_balanced', early_stop='logloss',
                                           role='robustness: G0 early-stopped on multi_logloss instead of macro-F1'),
    'G4_builtin_unweighted_loglossES': dict(objective='builtin', weights=None, early_stop='logloss',
                                            role='robustness: plain CE + log-loss ES + nested offsets (plug-in)'),
}
STATE = {}


# --------------------------------------------------------------------------- features
def latent_frames(fit_frame, others):
    from latent_v6 import LatentFeatures
    f = LatentFeatures()
    f.svd.set_params(n_components=128, random_state=SEED)
    xt = f.transform(fit_frame, fit=True)
    xs = [f.transform(o) for o in others]
    for col in ('category', 'rating'):
        dtype = pd.CategoricalDtype(sorted(set(xt[col])), ordered=False)
        xt[col] = pd.Categorical(xt[col], dtype=dtype)
        for x in xs:
            x[col] = pd.Categorical(x[col], dtype=dtype)       # unseen -> NaN (missing)
    return xt, xs


def build_outer(k):
    train, dev = STATE['train'], STATE['dev']
    tr, va = STATE['outer'][k]
    xt, (xv, xd) = latent_frames(train.iloc[tr], [train.iloc[va].drop(columns='spam_label'),
                                                    dev.drop(columns='spam_label')])
    return k, dict(X=xt, Xva=xv, Xdv=xd)


def build_inner(job):
    k, j = job
    train = STATE['train']
    tr, _ = STATE['outer'][k]
    itr, iva = STATE['inner'][k][j]
    sub = train.iloc[tr]
    xt, (xv,) = latent_frames(sub.iloc[itr], [sub.iloc[iva].drop(columns='spam_label')])
    return (k, j), dict(X=xt, Xva=xv)


# --------------------------------------------------------------------------- one booster
def class_weight_vector(cfg, y):
    if cfg['weights'] == 'sqrt_balanced':
        return sqrt_balanced_class_weights(y, K)
    if cfg['weights'] == 'effective':
        return effective_number_weights(y, K, cfg.get('beta', 0.999))
    return np.ones(K)


def own_loss(cfg, y_fit):
    w = class_weight_vector(cfg, y_fit)
    if cfg.get('logit_adjust_tau'):
        return LogitAdjustedCrossEntropy(log_class_prior(y_fit, K), tau=cfg['logit_adjust_tau'])
    if cfg['objective'] == 'focal' and cfg['gamma'] > 0:
        return FocalLoss(cfg['gamma'], class_weight=w)
    return SoftmaxCrossEntropy(class_weight=w)


def fit_booster(cfg, X, y, Xval=None, yval=None, rounds=None):
    """Returns (booster, n_rounds, offset) where predicted logits = raw + offset."""
    import lightgbm as lgb
    w = class_weight_vector(cfg, y)
    row_w = w[y]
    params = dict(PARAMS)
    train_init = None
    offset = np.zeros(K)
    if cfg['objective'] == 'focal':
        # start from the (weighted) log prior, as LightGBM's built-in multiclass does
        offset = np.log(np.bincount(y, weights=row_w, minlength=K) / row_w.sum())
        train_init = np.tile(offset, (len(y), 1))
        params['objective'] = make_lgb_focal_objective(cfg['gamma'], None, hess_floor=1e-6, hess_scale=K / (K - 1))
    else:
        params['objective'] = 'multiclass'
        if cfg.get('logit_adjust_tau'):
            # LA: training scores are z + tau log pi; predictions use z (offset stays 0)
            train_init = np.tile(cfg['logit_adjust_tau'] * log_class_prior(y, K), (len(y), 1))
    dtrain = lgb.Dataset(X, y, weight=row_w, init_score=train_init, categorical_feature=['category', 'rating'],
                         free_raw_data=False)
    if rounds is not None:
        booster = lgb.train(params, dtrain, num_boost_round=rounds)
        return booster, rounds, offset
    val_init = None if not np.any(offset) else np.tile(offset, (len(yval), 1))
    dval = lgb.Dataset(Xval, yval, init_score=val_init, reference=dtrain, categorical_feature=['category', 'rating'])
    feval = lgb_macro_f1_eval
    if cfg.get('early_stop') == 'logloss':          # built-in objectives only (raw scores need softmax)
        assert cfg['objective'] == 'builtin'
        params['metric'], feval = 'multi_logloss', None
    booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval], valid_names=['inner'],
                        feval=feval, callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)])
    return booster, int(booster.best_iteration), offset


def logits(booster, X, rounds, offset):
    raw = booster.predict(X, raw_score=True, num_iteration=rounds)
    out = np.asarray(raw, dtype=np.float64).reshape(len(X), K) + offset[None, :]
    if not np.isfinite(out).all():
        raise FloatingPointError('non-finite booster output')
    return out


def run_task(name, k):
    cfg = CONFIGS[name]
    y = STATE['y']
    tr, va = STATE['outer'][k]
    ytr, yva = y[tr], y[va]
    F = STATE['outer_feats'][k]
    t0 = time.time()
    Z = np.zeros((len(tr), K))
    best = []
    for j, (itr, iva) in enumerate(STATE['inner'][k]):
        Fi = STATE['inner_feats'][(k, j)]
        b, n, off = fit_booster(cfg, Fi['X'], ytr[itr], Fi['Xva'], ytr[iva])
        best.append(n)
        Z[iva] = logits(b, Fi['Xva'], n, off)
    bias = tune_class_bias(Z, ytr)
    temperature = fit_temperature(Z, ytr)
    rounds = max(1, int(round(np.mean(best))))
    b, _, off = fit_booster(cfg, F['X'], ytr, rounds=rounds)
    Ztr, Zva, Zdv = (logits(b, F[key], rounds, off) for key in ('X', 'Xva', 'Xdv'))
    loss = own_loss(cfg, ytr)
    pva = apply_bias(Zva, bias)
    rec = dict(config=name, fold=k, n_train=len(tr), n_val=len(va), inner_best_rounds=best, C_star=rounds,
               inner_oof_raw=macro_f1(ytr, Z.argmax(1)), inner_scores={str(rounds): macro_f1(ytr, apply_bias(Z, bias))},
               bias=bias.tolist(), temperature=temperature,
               train_loss=float(loss(Ztr, ytr)[0]), val_loss=float(loss(Zva, yva)[0]),
               val_nll_raw=nll(Zva, yva), val_nll_temperature=nll(Zva / temperature, yva),
               val_f1_raw=macro_f1(yva, Zva.argmax(1)), val_f1_post=macro_f1(yva, pva),
               val_per_class_post=per_class_f1(yva, pva).tolist(), seconds=time.time() - t0)
    return rec, Zva, Zdv


def safe_task(args):
    try:
        return run_task(*args)
    except Exception:
        return dict(config=args[0], fold=args[1], error=traceback.format_exc()), None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--inner-folds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS))
    ap.add_argument('--subsample', type=int, default=0, help='smoke test only')
    a = ap.parse_args()
    guard_not_macos()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, _, _, hashes = load(a.data_dir)
    if a.subsample:
        from sklearn.model_selection import train_test_split
        keep, _ = train_test_split(np.arange(len(train)), train_size=a.subsample, stratify=train.spam_label,
                                   random_state=a.seed)
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
    y = train.spam_label.to_numpy()
    outer = grouped_splits(train, y, a.outer_folds, a.seed)
    inner = {k: grouped_splits(train.iloc[tr], y[tr], a.inner_folds, a.seed) for k, (tr, _) in enumerate(outer)}
    fold_id = np.full(len(y), -1)
    for k, (_, va) in enumerate(outer):
        fold_id[va] = k
    STATE.update(train=train, dev=dev, y=y, ydev=dev.spam_label.to_numpy(), outer=outer, inner=inner, fold_id=fold_id)
    manifest = dict(data_sha256=hashes, seed=a.seed, outer_folds=a.outer_folds, inner_folds=a.inner_folds,
                    lightgbm_params=PARAMS, max_rounds=MAX_ROUNDS, patience=PATIENCE,
                    configs={n: CONFIGS[n] for n in a.configs},
                    source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))})
    todo = [(n, k) for n in a.configs for k in range(a.outer_folds) if not (out / f'{n}__fold{k}.json').exists()]
    target = out / 'manifest.json'
    if target.exists():       # never overwrite the manifest of earlier fits in this folder
        target = out / f"manifest_{'aggregation' if not todo else 'extra_' + '+'.join(sorted({n for n, _ in todo}))}.json"
    target.write_text(json.dumps(manifest, indent=2, default=str))
    t = time.time()
    if todo:
        with ProcessPoolExecutor(a.workers) as ex:
            STATE['outer_feats'] = dict(ex.map(build_outer, range(a.outer_folds)))
            STATE['inner_feats'] = dict(ex.map(build_inner, [(k, j) for k in range(a.outer_folds)
                                                             for j in range(a.inner_folds)]))
        print(f'latent features ready in {time.time() - t:.0f}s', flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            futures = {ex.submit(safe_task, job): job for job in todo}
            for fut in as_completed(futures):
                rec, Zva, Zdv = fut.result()
                if 'error' in rec:
                    print(f"ERROR {rec['config']} fold {rec['fold']}\n{rec['error']}", flush=True)
                    continue
                np.savez_compressed(out / f"{rec['config']}__fold{rec['fold']}.npz", Zva=Zva, Zdv=Zdv)
                (out / f"{rec['config']}__fold{rec['fold']}.json").write_text(json.dumps(rec, indent=2, default=float))
                print(f"{rec['config']} fold {rec['fold']}: rounds={rec['C_star']} raw={rec['val_f1_raw']:.4f} "
                      f"post={rec['val_f1_post']:.4f} ({rec['seconds']:.0f}s, elapsed {time.time() - t:.0f}s)", flush=True)
    done = [n for n in a.configs if all((out / f'{n}__fold{k}.json').exists() for k in range(a.outer_folds))]
    ra.STATE.update(STATE)
    refs = tuple(r for r in ('G0_builtin_multiclass_sqrtbal', 'G0LL_builtin_sqrtbal_loglossES') if r in done)
    table = ra.aggregate(out, done, configs=CONFIGS, refs=refs)
    (out / 'ablation_results.json').write_text(json.dumps(table, indent=2))
    md = ra.markdown(table, refs=refs, hyper='boosting rounds per fold')
    (out / 'ablation_table.md').write_text(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
