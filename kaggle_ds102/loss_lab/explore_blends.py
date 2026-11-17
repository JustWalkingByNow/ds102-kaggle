"""Exploratory (hypothesis-generating) analysis of saved OOF logits: do models trained
with different losses make complementary errors?  No fitting, no dev/test labels.

blend = mean_m softmax(Z_m / T_m,k)  with T_m,k the *nested* temperature of model m
in outer fold k (fitted on inner OOF).  Decision: raw argmax and cross-fit offsets.
Anything selected here must be confirmed on a fresh split seed before being trusted.

    python loss_lab/explore_blends.py --run loss_lab/output/ablation_seed102
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from losses import log_softmax, macro_f1  # noqa: E402
from postprocess import apply_bias, tune_class_bias  # noqa: E402


def load_model(run, name, n_folds):
    z = np.load(run / f'{name}__oof.npz')
    temps = [json.loads((run / f'{name}__fold{k}.json').read_text())['temperature'] for k in range(n_folds)]
    return z['oof_logits'], z['folds'], z['y'], np.asarray(temps)


def blend_logp(models, folds):
    out = np.zeros_like(models[0][0])
    for Z, _, _, temps in models:
        out += np.exp(log_softmax(Z / temps[folds][:, None]))
    return np.log(out / len(models))


def scores(logp, y, folds):
    raw = macro_f1(y, logp.argmax(1))
    cross = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        other = folds != k
        cross[folds == k] = apply_bias(logp[folds == k], tune_class_bias(logp[other], y[other]))
    per_fold_raw = [macro_f1(y[folds == k], logp[folds == k].argmax(1)) for k in np.unique(folds)]
    per_fold_cross = [macro_f1(y[folds == k], cross[folds == k]) for k in np.unique(folds)]
    return dict(raw=raw, crossfit=macro_f1(y, cross), fold_raw=per_fold_raw, fold_crossfit=per_fold_cross)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--base', default='B0_sq_hinge_balanced')
    a = ap.parse_args()
    run = Path(a.run)
    names = sorted(p.name.replace('__oof.npz', '') for p in run.glob('*__oof.npz'))
    models = {n: load_model(run, n, a.folds) for n in names}
    y, folds = models[a.base][2], models[a.base][1]
    rows = {}
    for n in names:
        rows[n] = scores(blend_logp([models[n]], folds), y, folds)
    for pair in itertools.combinations(names, 2):
        rows[' + '.join(pair)] = scores(blend_logp([models[p] for p in pair], folds), y, folds)
    (run / 'explore_blends.json').write_text(json.dumps(rows, indent=2))
    base = rows[a.base]
    print(f"{'model / blend (T-scaled prob average)':62s} {'raw':>7s} {'Δraw':>7s} {'crossfit':>8s} {'Δcross':>7s}")
    for key, r in sorted(rows.items(), key=lambda kv: -kv[1]['crossfit']):
        print(f"{key:62s} {r['raw']:7.4f} {r['raw'] - base['raw']:+7.4f} {r['crossfit']:8.4f} {r['crossfit'] - base['crossfit']:+7.4f}")


if __name__ == '__main__':
    main()
