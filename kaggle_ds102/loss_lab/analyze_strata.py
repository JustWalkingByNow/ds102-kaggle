"""Post-hoc analysis of saved OOF predictions (no fitting): the repo's class-1
template/novel strata, confusion matrices and per-class deltas.

Template mask exactly as experiment_v7.py / claude_review: a validation row is
'template' if its max token-set Jaccard to any class-1 comment in the training
part of its own outer fold is >= 0.6.  'Novel fake F1' = class-1 F1 on the
remaining rows (the repo's guard against template memorisation).

    python loss_lab/analyze_strata.py --data-dir <csv dir> --run loss_lab/output/ablation_seed102
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from data import SEED, REPO_SRC, grouped_splits, load  # noqa: E402

sys.path.insert(0, str(REPO_SRC / 'claude_review'))
from review_common import class1_strata, max_jaccard  # noqa: E402

K = 4


def template_mask(train, y, outer):
    mask = np.zeros(len(y), bool)
    for tr, va in outer:
        mask[va] = max_jaccard(train.comment.iloc[va].tolist(),
                               train.comment.iloc[tr[y[tr] == 1]].tolist()) >= .6
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--run', required=True, help='output folder of run_ablation / run_gbdt_ablation')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--outer-folds', type=int, default=5)
    a = ap.parse_args()
    run = Path(a.run)
    train, *_ = load(a.data_dir)
    y = train.spam_label.to_numpy()
    outer = grouped_splits(train, y, a.outer_folds, a.seed)
    mask = template_mask(train, y, outer)
    report = dict(template_rows=int(mask.sum()), class1_positives=int((y == 1).sum()),
                  class1_positives_with_template=int((mask & (y == 1)).sum()), configs={})
    for f in sorted(run.glob('*__oof.npz')):
        name = f.name.replace('__oof.npz', '')
        z = np.load(f)
        if not np.array_equal(z['y'], y):
            raise ValueError(f'{name}: label order mismatch')
        post = z['oof_post']
        cm = np.bincount(K * y + post, minlength=K * K).reshape(K, K)
        report['configs'][name] = dict(strata_post=class1_strata(y, post, mask),
                                       strata_raw=class1_strata(y, z['oof_logits'].argmax(1), mask),
                                       confusion_post=cm.tolist())
    (run / 'strata_report.json').write_text(json.dumps(report, indent=2))
    print(f"template rows {report['template_rows']}; class-1 positives with template "
          f"{report['class1_positives_with_template']}/{report['class1_positives']}")
    print(f"{'config':32s} {'novel F1':>9s} {'novel rec':>9s} {'templ F1':>9s} {'templ rec':>9s}")
    for name, r in report['configs'].items():
        s = r['strata_post']
        print(f"{name:32s} {s['novel']['f1']:9.4f} {s['novel']['recall']:9.4f} "
              f"{s['template']['f1']:9.4f} {s['template']['recall']:9.4f}")


if __name__ == '__main__':
    main()
