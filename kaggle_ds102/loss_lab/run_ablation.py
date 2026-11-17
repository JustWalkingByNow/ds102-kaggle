"""Controlled, fully nested K-fold ablation of loss functions for the DS102 linear text model.

Fixed across every row: data, outer/inner grouped splits (seed 102), FeaturesV2
features refitted per fold, linear model z = XW + b with zero init, full-batch
L-BFGS, and the same post-processing protocol.  Only the training loss changes.

Protocol (no validation statistic ever reaches a loss or a decision rule):
  outer: 5-fold StratifiedGroupKFold (groups = normalised comment)
  for every outer fold k and loss L:
    inner: 3-fold grouped split of the outer-train part, features refitted per inner fold
      -> inner OOF logits for every C in the family grid
      -> C* = argmax inner-OOF macro-F1 (after bias tuning), ties -> smaller C
      -> class offsets b_k and temperature T_k fitted on inner OOF at C*
    refit L with C* on the whole outer-train part, score outer fold k raw and with b_k
  dev: mean of the 5 outer models' logits + offsets tuned on pooled outer OOF (train labels only)

    python loss_lab/run_ablation.py --data-dir <csv dir> --out loss_lab/output/ablation_seed102
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
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from data import SEED, FoldFeatures, grouped_splits, guard_not_macos, load, nb_ratios, nb_views, sha256  # noqa: E402
from linear_model import LinearLossClassifier  # noqa: E402
from losses import SquaredHingeOvR, balanced_class_weights, macro_f1, nll, per_class_f1  # noqa: E402
from postprocess import apply_bias, fit_temperature, tune_class_bias  # noqa: E402

K = 4
SOFT_F1 = dict(name='soft_macro_f1')
# Pre-registered before any outer-validation label was read.  C grids were set
# from loss_lab/output/pilot_grid.json (inner split of outer-fold-0 training
# data only): each grid brackets that family's pilot optimum.
CONFIGS = {
    'R0_repo_v2_svm_nbsvm': dict(kind='repo_v2', role='as deployed: 0.75 LinearSVC + 0.25 NB-SVM, C=0.5'),
    'R1_repo_linearsvc': dict(kind='linearsvc', C=0.5, role='as deployed: sklearn LinearSVC, C=0.5'),
    'B0_sq_hinge_balanced': dict(kind='linear', spec=dict(name='squared_hinge_ovr', weights='balanced'),
                                 grid=[0.125, 0.25, 0.5], role='baseline loss (repo loss) in controlled trainer'),
    'B1_ce_balanced': dict(kind='linear', spec=dict(name='ce', weights='balanced'),
                           grid=[2.0, 8.0, 32.0], role='standard class-weighted cross-entropy'),
    'C1_cb_focal': dict(kind='linear', spec=dict(name='focal', gamma=2.0, weights='effective', beta=0.999),
                        grid=[8.0, 32.0, 128.0], role='candidate 1: class-balanced focal (g=2, b=0.999)'),
    'C2_logit_adjusted': dict(kind='linear', spec=dict(name='logit_adjusted', tau=1.0),
                              grid=[2.0, 8.0, 32.0], role='candidate 2: logit-adjusted CE (tau=1)'),
    'C3_composite_ce_softf1': dict(kind='linear', grid=[2.0, 8.0, 32.0],
                                   spec=dict(name='composite', base=dict(name='ce', weights='balanced'),
                                             metric=SOFT_F1, stages=4, alpha_min=0.25),
                                   role='composite: annealed CE_bal -> soft macro-F1'),
    'C4_composite_la_softf1': dict(kind='linear', grid=[2.0, 8.0, 32.0],
                                   spec=dict(name='composite', base=dict(name='logit_adjusted', tau=1.0),
                                             metric=SOFT_F1, stages=4, alpha_min=0.25),
                                   role='composite: annealed LA-CE -> soft macro-F1'),
}
STATE = {}


# --------------------------------------------------------------------------- features
def build_outer(k):
    train, dev, y = STATE['train'], STATE['dev'], STATE['y']
    tr, va = STATE['outer'][k]
    ff = FoldFeatures(train.iloc[tr])
    Xva, tva, mva = ff.transform(train.iloc[va].drop(columns='spam_label'))
    Xdv, tdv, mdv = ff.transform(dev.drop(columns='spam_label'))
    return k, dict(X=ff.X, text=ff.text, meta=ff.meta, Xva=Xva, tva=tva, mva=mva, Xdv=Xdv, tdv=tdv, mdv=mdv)


def build_inner(job):
    k, j = job
    train = STATE['train']
    tr, _ = STATE['outer'][k]
    itr, iva = STATE['inner'][k][j]
    sub = train.iloc[tr]
    ff = FoldFeatures(sub.iloc[itr])
    Xiv, tiv, miv = ff.transform(sub.iloc[iva].drop(columns='spam_label'))
    return (k, j), dict(X=ff.X, text=ff.text, meta=ff.meta, Xva=Xiv, tva=tiv, mva=miv)


# --------------------------------------------------------------------------- model families
def repo_v2_scores(text, meta, y, eval_views):
    """Exact repo v2 recipe (train_submit_v2.RECIPE) on precomputed fold features."""
    from sklearn.svm import LinearSVC
    from train_submit import make_model
    from scipy import sparse
    X = sparse.hstack([text, meta], format='csr')
    svm = make_model({'C': .5, 'weight': 'balanced'}, y).fit(X, y)
    if np.max(svm.n_iter_) >= svm.max_iter:
        raise RuntimeError('SVM did not converge')
    out = [0.75 * svm.decision_function(sparse.hstack([t, m], format='csr')) for t, m in eval_views]
    ratios = nb_ratios(text, y)
    for c, r in enumerate(ratios):
        nb = LinearSVC(C=.5, class_weight='balanced', dual=True, max_iter=30000, random_state=SEED)
        nb.fit(nb_views(text, meta, [r])[0], (y == c).astype(int))
        if np.max(nb.n_iter_) >= nb.max_iter:
            raise RuntimeError('NB-SVM did not converge')
        for i, (t, m) in enumerate(eval_views):
            out[i][:, c] += 0.25 * nb.decision_function(nb_views(t, m, [r])[0])
    return out, None


def linearsvc_scores(X, y, eval_X, C):
    from train_submit import make_model
    m = make_model({'C': C, 'weight': 'balanced'}, y).fit(X, y)
    if np.max(m.n_iter_) >= m.max_iter:
        raise RuntimeError('LinearSVC did not converge')
    loss = SquaredHingeOvR(balanced_class_weights(y, K))
    info = dict(train_loss=float(loss(m.decision_function(X), y)[0]), loss_obj=loss)
    return [m.decision_function(Z) for Z in eval_X], info


# --------------------------------------------------------------------------- one (config, fold) task
def run_task(name, k):
    cfg = CONFIGS[name]
    y = STATE['y']
    tr, va = STATE['outer'][k]
    ytr, yva = y[tr], y[va]
    F = STATE['outer_feats'][k]
    t0 = time.time()
    rec = dict(config=name, fold=k, n_train=len(tr), n_val=len(va))
    inner_oof = {}
    grid = cfg.get('grid', [cfg.get('C', 0.5)])
    for C in grid:
        Z = np.zeros((len(tr), K))
        for j, (_, iva) in enumerate(STATE['inner'][k]):
            Fi = STATE['inner_feats'][(k, j)]
            iy = ytr[STATE['inner'][k][j][0]]
            if cfg['kind'] == 'linear':
                m = LinearLossClassifier(cfg['spec'], C=C).fit(Fi['X'], iy)
                Z[iva] = m.decision_function(Fi['Xva'])
            elif cfg['kind'] == 'linearsvc':
                Z[iva] = linearsvc_scores(Fi['X'], iy, [Fi['Xva']], C)[0][0]
            else:
                Z[iva] = repo_v2_scores(Fi['text'], Fi['meta'], iy, [(Fi['tva'], Fi['mva'])])[0][0]
        inner_oof[C] = Z
    scores = {}
    for C, Z in inner_oof.items():
        scores[C] = macro_f1(ytr, apply_bias(Z, tune_class_bias(Z, ytr)))
    best = max(scores.values())
    C_star = min(C for C, s in scores.items() if s >= best - 1e-12)
    bias = tune_class_bias(inner_oof[C_star], ytr)
    temperature = fit_temperature(inner_oof[C_star], ytr)
    rec.update(inner_scores={str(C): s for C, s in scores.items()}, C_star=C_star,
               inner_oof_raw=macro_f1(ytr, inner_oof[C_star].argmax(1)), bias=bias.tolist(), temperature=temperature)
    # ---- refit on the whole outer-train part with C*, score outer fold k and dev
    if cfg['kind'] == 'linear':
        m = LinearLossClassifier(cfg['spec'], C=C_star).fit(F['X'], ytr)
        Zva, Zdv = m.decision_function(F['Xva']), m.decision_function(F['Xdv'])
        rec.update(train_loss=m.train_loss_, val_loss=m.loss_on(F['Xva'], yva),
                   stages=m.stages_, fit_seconds=m.fit_seconds_)
    elif cfg['kind'] == 'linearsvc':
        (Zva, Zdv), info = linearsvc_scores(F['X'], ytr, [F['Xva'], F['Xdv']], C_star)
        rec.update(train_loss=info['train_loss'], val_loss=float(info['loss_obj'](Zva, yva)[0]))
    else:
        (Zva, Zdv), _ = repo_v2_scores(F['text'], F['meta'], ytr, [(F['tva'], F['mva']), (F['tdv'], F['mdv'])])
        rec.update(train_loss=None, val_loss=None)
    pva = apply_bias(Zva, bias)
    rec.update(val_nll_raw=nll(Zva, yva), val_nll_temperature=nll(Zva / temperature, yva),
               val_f1_raw=macro_f1(yva, Zva.argmax(1)), val_f1_post=macro_f1(yva, pva),
               val_per_class_post=per_class_f1(yva, pva).tolist(), seconds=time.time() - t0)
    return rec, Zva, Zdv


def safe_task(args):
    try:
        return run_task(*args)
    except Exception:
        return dict(config=args[0], fold=args[1], error=traceback.format_exc()), None, None


# --------------------------------------------------------------------------- aggregation
def _f1_from_cells(cells):
    cm = cells.reshape(-1, K, K)
    den = cm.sum(axis=1) + cm.sum(axis=2)
    diag = np.einsum('bii->bi', cm)
    return np.mean(np.divide(2 * diag, den, out=np.zeros_like(den, dtype=float), where=den > 0), axis=1)


def paired_bootstrap(y, pred_a, pred_b, groups, n_boot=2000, seed=SEED):
    """Cluster bootstrap (groups = normalised comment) of macro-F1(b) - macro-F1(a).

    Each group's confusion cells are summed once; a replicate is a multinomial
    re-draw of groups, so the whole bootstrap is one matrix product.
    """
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    cells_a = np.zeros((G, K * K))
    cells_b = np.zeros((G, K * K))
    np.add.at(cells_a, (inv, K * y + pred_a), 1.0)
    np.add.at(cells_b, (inv, K * y + pred_b), 1.0)
    counts = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n_boot)]).astype(float)
    deltas = _f1_from_cells(counts @ cells_b) - _f1_from_cells(counts @ cells_a)
    return dict(mean=float(deltas.mean()), q025=float(np.quantile(deltas, .025)),
                q975=float(np.quantile(deltas, .975)), p_gt_0=float((deltas > 0).mean()))


LINEAR_REFS = ('B0_sq_hinge_balanced', 'R0_repo_v2_svm_nbsvm')


def aggregate(out, names, configs=None, refs=LINEAR_REFS):
    configs = CONFIGS if configs is None else configs
    y, ydev, groups = STATE['y'], STATE['ydev'], STATE['train'].comment.to_numpy()
    folds = STATE['fold_id']
    table, preds, preds_cross = {}, {}, {}
    for name in names:
        recs = [json.loads((out / f'{name}__fold{k}.json').read_text()) for k in range(len(STATE['outer']))]
        z = {k: np.load(out / f'{name}__fold{k}.npz') for k in range(len(STATE['outer']))}
        oof = np.zeros((len(y), K))
        post = np.zeros(len(y), dtype=int)
        for k, (_, va) in enumerate(STATE['outer']):
            oof[va] = z[k]['Zva']
            post[va] = apply_bias(z[k]['Zva'], recs[k]['bias'])
        # scale-matched variant: offsets for fold k tuned on the OOF of the *other*
        # outer folds (same-size models; fold k's labels never used)
        cross = np.zeros(len(y), dtype=int)
        for k, (_, va) in enumerate(STATE['outer']):
            other = folds != k
            cross[va] = apply_bias(oof[va], tune_class_bias(oof[other], y[other]))
        f_cross = np.array([macro_f1(y[folds == k], cross[folds == k]) for k in range(len(STATE['outer']))])
        dev_logits = np.mean([z[k]['Zdv'] for k in z], axis=0)
        dev_bias = tune_class_bias(oof, y)                      # train OOF labels only
        dev_post = apply_bias(dev_logits, dev_bias)
        preds[name], preds_cross[name] = post, cross
        np.savez_compressed(out / f'{name}__oof.npz', oof_logits=oof, oof_post=post, oof_crossfit=cross, folds=folds,
                            dev_logits=dev_logits, dev_bias=dev_bias, y=y)
        col = lambda key: np.array([r[key] for r in recs], dtype=float)  # noqa: E731
        f_raw, f_post = col('val_f1_raw'), col('val_f1_post')
        table[name] = dict(
            role=configs[name]['role'], C_star=[r['C_star'] for r in recs],
            train_loss=None if recs[0]['train_loss'] is None else [float(col('train_loss').mean()), float(col('train_loss').std())],
            val_loss=None if recs[0]['val_loss'] is None else [float(col('val_loss').mean()), float(col('val_loss').std())],
            val_nll_raw=[float(col('val_nll_raw').mean()), float(col('val_nll_raw').std())],
            val_nll_temperature=[float(col('val_nll_temperature').mean()), float(col('val_nll_temperature').std())],
            fold_f1_raw=[float(f_raw.mean()), float(f_raw.std())], fold_f1_post=[float(f_post.mean()), float(f_post.std())],
            fold_f1_post_values=f_post.tolist(), fold_f1_raw_values=f_raw.tolist(),
            fold_f1_crossfit=[float(f_cross.mean()), float(f_cross.std())], fold_f1_crossfit_values=f_cross.tolist(),
            pooled_oof_f1_crossfit=macro_f1(y, cross),
            pooled_oof_f1_raw=macro_f1(y, oof.argmax(1)), pooled_oof_f1_post=macro_f1(y, post),
            pooled_per_class_post=per_class_f1(y, post).tolist(),
            dev_f1_raw=macro_f1(ydev, dev_logits.argmax(1)), dev_f1_post=macro_f1(ydev, dev_post),
            dev_per_class_post=per_class_f1(ydev, dev_post).tolist(),
            seconds=float(col('seconds').sum()))
    for ref in refs:
        if ref not in preds:
            continue
        for name in names:
            if name == ref:
                continue
            fa = np.array(table[ref]['fold_f1_post_values'])
            fb = np.array(table[name]['fold_f1_post_values'])
            table[name][f'vs_{ref}'] = dict(fold_deltas=(fb - fa).tolist(), folds_positive=int((fb > fa).sum()),
                                            bootstrap=paired_bootstrap(y, preds[ref], preds[name], groups))
            ca = np.array(table[ref]['fold_f1_crossfit_values'])
            cb = np.array(table[name]['fold_f1_crossfit_values'])
            table[name][f'vs_{ref}_crossfit'] = dict(fold_deltas=(cb - ca).tolist(), folds_positive=int((cb > ca).sum()),
                                                     bootstrap=paired_bootstrap(y, preds_cross[ref], preds_cross[name], groups))
    return table


def markdown(table, refs=LINEAR_REFS, hyper='C* per fold'):
    lines = [f'| Config | Role | {hyper} | Train loss | Val loss | Val NLL (raw / T-scaled) | '
             'OOF F1 raw (fold mean±std) | OOF F1 nested-post (fold mean±std) | Pooled OOF F1 nested-post | '
             'OOF F1 cross-fit post (fold mean±std) / pooled | '
             'Per-class F1 nested-post [0,1,2,3] | Dev F1 raw / post |', '|' + '---|' * 12]
    fmt = lambda v: '—' if v is None else f'{v[0]:.4f} ± {v[1]:.4f}'  # noqa: E731
    for name, r in table.items():
        lines.append(
            f"| {name} | {r['role']} | {', '.join(f'{c:g}' for c in r['C_star'])} | {fmt(r['train_loss'])} | "
            f"{fmt(r['val_loss'])} | {r['val_nll_raw'][0]:.4f} / {r['val_nll_temperature'][0]:.4f} | "
            f"{fmt(r['fold_f1_raw'])} | {fmt(r['fold_f1_post'])} | {r['pooled_oof_f1_post']:.4f} | "
            f"{fmt(r['fold_f1_crossfit'])} / {r['pooled_oof_f1_crossfit']:.4f} | "
            f"{', '.join(f'{v:.3f}' for v in r['pooled_per_class_post'])} | {r['dev_f1_raw']:.4f} / {r['dev_f1_post']:.4f} |")
    short = [ref.split('_')[0] for ref in refs]
    for suffix, title in (('', 'nested offsets (primary, strict)'), ('_crossfit', 'cross-fit offsets (scale-matched)')):
        head = ' | '.join(f'Δ vs {s} per fold | folds > {s} | bootstrap Δ mean [95% CI], P(Δ>0)' for s in short)
        lines += ['', f'Paired comparison — {title}:', '', f'| Config | {head} |', '|' + '---|' * (1 + 3 * len(refs))]
        for name, r in table.items():
            cells = []
            for ref in refs:
                c = r.get(f'vs_{ref}{suffix}')
                if c is None:
                    cells += ['—', '—', '—']
                else:
                    b = c['bootstrap']
                    cells += [', '.join(f'{d:+.4f}' for d in c['fold_deltas']), f"{c['folds_positive']}/{len(c['fold_deltas'])}",
                              f"{b['mean']:+.4f} [{b['q025']:+.4f}, {b['q975']:+.4f}], {b['p_gt_0']:.2f}"]
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED, help='split seed (models are deterministic)')
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--inner-folds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS))
    ap.add_argument('--subsample', type=int, default=0, help='smoke test: stratified subsample of train rows')
    ap.add_argument('--grid-limit', type=int, default=0, help='smoke test: keep only the first N grid values')
    a = ap.parse_args()
    guard_not_macos()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, _, _, hashes = load(a.data_dir)
    if a.subsample:
        from sklearn.model_selection import train_test_split
        keep, _ = train_test_split(np.arange(len(train)), train_size=a.subsample,
                                   stratify=train.spam_label, random_state=a.seed)
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
    if a.grid_limit:
        for cfg in CONFIGS.values():
            if 'grid' in cfg:
                cfg['grid'] = cfg['grid'][:a.grid_limit]
    y = train.spam_label.to_numpy()
    outer = grouped_splits(train, y, a.outer_folds, a.seed)
    inner = {k: grouped_splits(train.iloc[tr], y[tr], a.inner_folds, a.seed) for k, (tr, _) in enumerate(outer)}
    fold_id = np.full(len(y), -1)
    for k, (_, va) in enumerate(outer):
        fold_id[va] = k
    STATE.update(train=train, dev=dev, y=y, ydev=dev.spam_label.to_numpy(), outer=outer, inner=inner, fold_id=fold_id)
    manifest = dict(data_sha256=hashes, seed=a.seed, outer_folds=a.outer_folds, inner_folds=a.inner_folds,
                    configs={n: CONFIGS[n] for n in a.configs},
                    source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
                    protocol=__doc__.split('Protocol')[1].split('python loss_lab')[0].strip())
    todo = [(n, k) for n in a.configs for k in range(a.outer_folds) if not (out / f'{n}__fold{k}.json').exists()]
    # keep the manifest of the code that fitted the models; a pure re-aggregation gets its own file
    (out / ('manifest.json' if todo else 'manifest_aggregation.json')).write_text(json.dumps(manifest, indent=2))
    t = time.time()
    if todo:
        with ProcessPoolExecutor(a.workers) as ex:
            STATE['outer_feats'] = dict(ex.map(build_outer, range(a.outer_folds)))
            STATE['inner_feats'] = dict(ex.map(build_inner, [(k, j) for k in range(a.outer_folds) for j in range(a.inner_folds)]))
        print(f'features ready in {time.time() - t:.0f}s', flush=True)
        cost = {'C3_composite_ce_softf1': 4, 'C1_cb_focal': 2, 'B1_ce_balanced': 2, 'C2_logit_adjusted': 2}
        todo.sort(key=lambda job: -cost.get(job[0], 1))
        with ProcessPoolExecutor(a.workers) as ex:      # fork: workers share the feature matrices
            futures = {ex.submit(safe_task, job): job for job in todo}
            for fut in as_completed(futures):
                rec, Zva, Zdv = fut.result()
                name, k = rec['config'], rec['fold']
                if 'error' in rec:
                    print(f'ERROR {name} fold {k}\n{rec["error"]}', flush=True)
                    continue
                np.savez_compressed(out / f'{name}__fold{k}.npz', Zva=Zva, Zdv=Zdv)
                (out / f'{name}__fold{k}.json').write_text(json.dumps(rec, indent=2, default=float))
                print(f"{name} fold {k}: C*={rec['C_star']} raw={rec['val_f1_raw']:.4f} post={rec['val_f1_post']:.4f} "
                      f"({rec['seconds']:.0f}s, elapsed {time.time() - t:.0f}s)", flush=True)
    done = [n for n in a.configs if all((out / f'{n}__fold{k}.json').exists() for k in range(a.outer_folds))]
    table = aggregate(out, done)
    (out / 'ablation_results.json').write_text(json.dumps(table, indent=2))
    (out / 'ablation_table.md').write_text(markdown(table) + '\n')
    print(markdown(table))


if __name__ == '__main__':
    main()
