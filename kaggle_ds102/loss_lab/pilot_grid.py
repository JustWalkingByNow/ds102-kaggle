"""Pilot: choose the C grid range and measure cost WITHOUT touching outer-validation labels.

Uses only the training part of outer fold 0 (5-fold grouped split, seed 102),
split again 3-fold; fits on inner-train 0, scores inner-val 0.  The outer
validation fold of the ablation is never read here.

    python loss_lab/pilot_grid.py --data-dir <csv dir> --out <dir>
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.svm import LinearSVC  # noqa: E402

from data import SEED, FoldFeatures, grouped_splits, guard_not_macos, load  # noqa: E402
from linear_model import LinearLossClassifier  # noqa: E402
from losses import macro_f1  # noqa: E402
from postprocess import apply_bias, tune_class_bias  # noqa: E402

SPECS = {
    'sq_hinge_balanced': dict(name='squared_hinge_ovr', weights='balanced'),
    'ce_balanced': dict(name='ce', weights='balanced'),
    'cb_focal': dict(name='focal', gamma=2.0, weights='effective', beta=0.999),
    'logit_adjusted': dict(name='logit_adjusted', tau=1.0),
    'composite': dict(name='composite', base=dict(name='ce', weights='balanced'),
                      metric=dict(name='soft_macro_f1'), stages=4, alpha_min=0.25),
}
GRID = [0.125, 0.5, 2.0, 8.0, 32.0]
STATE = {}


def run(job):
    name, C = job
    Xtr, ytr, Xva, yva = STATE['Xtr'], STATE['ytr'], STATE['Xva'], STATE['yva']
    t = time.time()
    m = LinearLossClassifier(SPECS[name], C=C).fit(Xtr, ytr)
    Z = m.decision_function(Xva)
    b = tune_class_bias(Z, yva)        # in-sample bias: optimistic, only for range-finding
    return dict(loss=name, C=C, seconds=round(time.time() - t, 1),
                iters=[s['n_iter'] for s in m.stages_], converged=[s['converged'] for s in m.stages_],
                raw_f1=round(macro_f1(yva, Z.argmax(1)), 5), biased_f1_in_sample=round(macro_f1(yva, apply_bias(Z, b)), 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    guard_not_macos()
    train, *_ = load(a.data_dir)
    y = train.spam_label.to_numpy()
    otr, _ = grouped_splits(train, y, 5, SEED)[0]
    sub = train.iloc[otr].reset_index(drop=True)
    ys = y[otr]
    itr, iva = grouped_splits(sub, ys, 3, SEED)[0]
    t = time.time()
    ff = FoldFeatures(sub.iloc[itr])
    Xva, _, _ = ff.transform(sub.iloc[iva].drop(columns='spam_label'))
    STATE.update(Xtr=ff.X, ytr=ys[itr], Xva=Xva, yva=ys[iva])
    print('features', ff.X.shape, round(time.time() - t, 1), 's', flush=True)
    t = time.time()
    svc = LinearSVC(C=0.5, class_weight='balanced', dual=True, max_iter=30000, random_state=SEED).fit(ff.X, ys[itr])
    Zs = svc.decision_function(Xva)
    rows = [dict(loss='sklearn_LinearSVC_C0.5', seconds=round(time.time() - t, 1), raw_f1=round(macro_f1(ys[iva], Zs.argmax(1)), 5),
                 biased_f1_in_sample=round(macro_f1(ys[iva], apply_bias(Zs, tune_class_bias(Zs, ys[iva]))), 5))]
    print(rows[-1], flush=True)
    jobs = [(n, C) for n in SPECS for C in GRID]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run, jobs):
            rows.append(r)
            print(r, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pilot_grid.json').write_text(json.dumps(dict(
        protocol='inner split 0 of outer-fold-0 training part; outer validation never read', grid=GRID,
        specs=SPECS, rows=rows), indent=2))


if __name__ == '__main__':
    main()
