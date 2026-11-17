"""Transfer check: logit adjustment in the repo's production latent-CatBoost branch.

Same five outer folds as run_ablation.py (seed 102).  Features: latent_v6.LatentFeatures
with 128 SVD components refitted per fold (the V11 setting).  Model: exactly
latent_v6.make_model() (800 trees, depth 5, lr 0.06, l2 8, MultiClass) except:

  K0  auto_class_weights='SqrtBalanced'                  (production recipe)
  K1  no class weights + training Pool baseline = tau*log(pi), tau = 1   (logit adjustment)

Predictions are RawFormulaVal logits *without* the baseline.  One fit per fold and
config (fixed 800 trees, no early stopping); decision rules: raw argmax and
cross-fit offsets (fold k's offsets tuned on the other folds' OOF only).

    python loss_lab/run_catboost_la.py --data-dir <csv dir> --out loss_lab/output/catboost_la_seed102
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from data import SEED, grouped_splits, guard_not_macos, load, sha256  # noqa: E402
from losses import log_class_prior, macro_f1, nll, per_class_f1  # noqa: E402
from postprocess import apply_bias, tune_class_bias  # noqa: E402
from run_ablation import paired_bootstrap  # noqa: E402

K = 4
CONFIGS = {'K0_catboost_sqrtbalanced': dict(la=False, role='production latent CatBoost (SqrtBalanced)'),
           'K1_catboost_logit_adjusted': dict(la=True, tau=1.0, role='same model, logit-adjusted baseline, no weights')}
STATE = {}


def run_fold(job):
    name, k = job
    from catboost import Pool
    from latent_v6 import LatentFeatures, make_model
    train, dev, y = STATE['train'], STATE['dev'], STATE['y']
    tr, va = STATE['outer'][k]
    t0 = time.time()
    f = LatentFeatures()
    f.svd.set_params(n_components=128, random_state=SEED)
    xt = f.transform(train.iloc[tr], fit=True)
    xv = f.transform(train.iloc[va].drop(columns='spam_label'))
    xd = f.transform(dev.drop(columns='spam_label'))
    from catboost import CatBoostClassifier
    params = make_model().get_params()
    params['thread_count'] = 1
    cfg = CONFIGS[name]
    if cfg['la']:
        params.pop('auto_class_weights')         # CatBoost cannot parse a null value; drop the key
    m = CatBoostClassifier(**params)
    if cfg['la']:
        base = np.tile(cfg['tau'] * log_class_prior(y[tr], K), (len(tr), 1))
        m.fit(Pool(xt, y[tr], cat_features=['category', 'rating'], baseline=base), verbose=False)
    else:
        m.fit(Pool(xt, y[tr], cat_features=['category', 'rating']), verbose=False)
    Zva = m.predict(Pool(xv, cat_features=['category', 'rating']), prediction_type='RawFormulaVal')
    Zdv = m.predict(Pool(xd, cat_features=['category', 'rating']), prediction_type='RawFormulaVal')
    return name, k, np.asarray(Zva, float), np.asarray(Zdv, float), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    guard_not_macos()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, _, _, hashes = load(a.data_dir)
    y = train.spam_label.to_numpy()
    outer = grouped_splits(train, y, 5, a.seed)
    folds = np.full(len(y), -1)
    for k, (_, va) in enumerate(outer):
        folds[va] = k
    STATE.update(train=train, dev=dev, y=y, outer=outer)
    (out / 'manifest.json').write_text(json.dumps(dict(
        data_sha256=hashes, seed=a.seed, configs=CONFIGS,
        source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}), indent=2))
    jobs = [(n, k) for n in CONFIGS for k in range(5) if not (out / f'{n}__fold{k}.npz').exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        for name, k, Zva, Zdv, sec in ex.map(run_fold, jobs):
            np.savez_compressed(out / f'{name}__fold{k}.npz', Zva=Zva, Zdv=Zdv, seconds=sec)
            print(f'{name} fold {k}: raw={macro_f1(y[outer[k][1]], Zva.argmax(1)):.4f} ({sec:.0f}s)', flush=True)
    ydev = dev.spam_label.to_numpy()
    table, preds = {}, {}
    for name in CONFIGS:
        oof = np.zeros((len(y), K))
        zd = []
        for k, (_, va) in enumerate(outer):
            z = np.load(out / f'{name}__fold{k}.npz')
            oof[va] = z['Zva']
            zd.append(z['Zdv'])
        cross = np.zeros(len(y), dtype=int)
        for k in range(5):
            other = folds != k
            cross[folds == k] = apply_bias(oof[folds == k], tune_class_bias(oof[other], y[other]))
        dev_logits = np.mean(zd, axis=0)
        dev_post = apply_bias(dev_logits, tune_class_bias(oof, y))
        preds[name] = cross
        fr = [macro_f1(y[folds == k], oof[folds == k].argmax(1)) for k in range(5)]
        fc = [macro_f1(y[folds == k], cross[folds == k]) for k in range(5)]
        table[name] = dict(role=CONFIGS[name]['role'], fold_f1_raw=fr, fold_f1_crossfit=fc,
                           mean_raw=float(np.mean(fr)), std_raw=float(np.std(fr)),
                           mean_crossfit=float(np.mean(fc)), std_crossfit=float(np.std(fc)),
                           pooled_crossfit=macro_f1(y, cross), per_class_crossfit=per_class_f1(y, cross).tolist(),
                           val_nll_raw=nll(oof, y), dev_f1_raw=macro_f1(ydev, dev_logits.argmax(1)),
                           dev_f1_post=macro_f1(ydev, dev_post))
    a0, a1 = list(CONFIGS)
    d = np.array(table[a1]['fold_f1_crossfit']) - np.array(table[a0]['fold_f1_crossfit'])
    table['K1_vs_K0'] = dict(fold_deltas_crossfit=d.tolist(), folds_positive=int((d > 0).sum()),
                             bootstrap_crossfit=paired_bootstrap(y, preds[a0], preds[a1], train.comment.to_numpy()))
    (out / 'results.json').write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
