"""Data, splits and fold-local features for loss_lab (reuses the repo's own code).

Features are exactly the repo's ``FeaturesV2`` (word 1-3g + char 2-5g TF-IDF,
product/category/rating/dense meta), refitted on the *training part of every
fold*; validation/dev/test rows are only transformed.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parent
for p in (str(HERE), str(REPO_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from improved_features import FeaturesV2, nb_ratios, prepare  # noqa: E402
from train_submit import load_data  # noqa: E402

SEED = 102
FILES = ['ds102_train.csv', 'ds102_dev.csv', 'ds102_test.csv', 'ds102_samplesubmission.csv']


def guard_not_macos():
    """The project owner forbids fitting on their Mac (overheating); Linux hosts
    such as hosted Colab or a cloud container are fine."""
    if sys.platform == 'darwin' and not os.environ.get('LOSS_LAB_ALLOW_MAC'):
        raise RuntimeError('Refusing to fit on macOS: run on hosted Colab or another Linux host.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(data_dir):
    data_dir = Path(data_dir)
    data = load_data(data_dir)
    train, dev, test = (prepare(data[k]) for k in ('train', 'dev', 'test'))
    hashes = {f: sha256(data_dir / f) for f in FILES}
    return train, dev, test, data['samplesubmission'], hashes


def grouped_splits(frame, y, n_splits, seed=SEED):
    """StratifiedGroupKFold grouped by normalised comment (repo protocol)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in splitter.split(frame, y, groups=frame.comment.to_numpy())]


class FoldFeatures:
    """Fit FeaturesV2 on one training part; transform any other frames."""

    def __init__(self, fit_frame):
        self.features = FeaturesV2()
        text, meta = self.features.fit_transform(fit_frame)
        self.text, self.meta = text.astype(np.float64), meta.astype(np.float64)
        self.X = sparse.hstack([self.text, self.meta], format='csr')

    def transform(self, frame):
        text, meta = self.features.transform(frame)
        text, meta = text.astype(np.float64), meta.astype(np.float64)
        return sparse.hstack([text, meta], format='csr'), text, meta


def nb_views(text, meta, ratios):
    """NB-SVM inputs (repo v2): text * r_c stacked with meta, one per class."""
    return [sparse.hstack([text.multiply(r).tocsr(), meta], format='csr') for r in ratios]


__all__ = ['SEED', 'FoldFeatures', 'grouped_splits', 'guard_not_macos', 'load', 'nb_ratios', 'nb_views', 'sha256']
