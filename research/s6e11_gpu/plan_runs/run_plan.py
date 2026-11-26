"""Run the S6E11 plan experiments on a local NVIDIA GPU and checkpoint each fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parent
CONTROL_PATH = REPO / 'configs' / 'phase8_controls.json'
TARGET = 'is_fraud'

sys.path.insert(0, str(REPO))
from src.validation import assign_folds, early_stop_assignments, partition_for_fold, smoke_sample


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    groups: tuple[str, ...]
    parent: str = ''
    smoothing: float = 20.0
    model_seed: int = 42
    encoder_seed: int = 2026
    params: dict = field(default_factory=dict)
    boost: bool = False
    rounds_factor: float = 1.0


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_npz(path, **arrays):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, **arrays)
    temp.replace(path)


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def metric_row(y, prediction):
    return {
        'oof_auc': float(roc_auc_score(y, prediction)),
        'average_precision': float(average_precision_score(y, prediction)),
        'log_loss': float(log_loss(y, prediction)),
    }


def get_specs():
    references = read_json(CONTROL_PATH)
    pairs = []
    for label, family, raw, control in (
        ('T01', 'glm', 'glm_raw', 'GPU-B02'),
        ('T02', 'lgb', 'raw', 'GPU-S6-L04'),
        ('T03', 'xgb', 'raw', 'GPU-S6-X02'),
    ):
        params = references[label]['params'].copy()
        base = Experiment(control, family, (raw,), params=params)
        candidate = replace(base, name=label, groups=(raw, 'te13'), parent=control)
        pairs.append((base, candidate))
    return pairs


class PlanRun:
    def __init__(self, output, *, smoke=False):
        from features import FeatureBuilder

        self.output = Path(output).resolve()
        self.smoke = smoke
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = read_json(REPO / 'configs' / 'default.json')
        references = read_json(CONTROL_PATH)['T01']
        self.input_hashes = {name: sha256(PROJECT.parent / "data" / name) for name in references['input_files']}
        for name, digest in self.input_hashes.items():
            if not smoke and digest != references['input_files'][name]['sha256']:
                raise ValueError(f'{name} differs from the plan data snapshot')
        self.train, self.test, self.sample = (
            pd.read_csv(PROJECT.parent / "data" / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv')
        )
        if smoke:
            self.train, self.test, self.sample = smoke_sample(self.train, self.test, self.sample, self.config)
        self.y = self.train[TARGET].to_numpy(dtype=np.int8)
        self.ids = self.train['id'].to_numpy()
        self.test_ids = self.test['id'].to_numpy()
        if (not np.array_equal(self.test_ids, self.sample['id'].to_numpy())
                or not self.train['id'].is_unique or not self.test['id'].is_unique
                or np.intersect1d(self.ids, self.test_ids).size or TARGET in self.test):
            raise ValueError('Train, test or sample ID contract violated')
        self.test_indices = np.arange(len(self.test_ids), dtype=np.int64) + len(self.y)
        n_folds = 2 if smoke else self.config['n_folds']
        folds = assign_folds(self.train, TARGET, n_folds, self.config['seed'])
        early = early_stop_assignments(self.train, folds, self.config, n_folds)
        self.fold_hash, self.early_hash = frame_hash(folds), frame_hash(early)
        if not smoke and (self.fold_hash != references['fold_hash']
                          or self.early_hash != references['early_stop_hash']):
            raise ValueError('Assignment differs from the plan fold/early-stop snapshot')
        self.folds = folds['fold'].to_numpy()
        self.partitions = [partition_for_fold(self.train, folds, early, self.config, fold)
                           for fold in range(n_folds)]
        self.builder = FeatureBuilder(self.train, self.test)
        self.manifest = {
            'smoke': smoke, 'rows': len(self.y), 'test_rows': len(self.test),
            'fold_hash': self.fold_hash, 'early_stop_hash': self.early_hash,
            'inputs': self.input_hashes, 'models_use_llm': False,
        }
        self.reports = {}
        manifest_path = self.output / 'manifest.json'
        if manifest_path.exists():
            if read_json(manifest_path) != self.manifest:
                raise ValueError(f'Results directory contains a different input manifest: {self.output}')
        else:
            write_json(manifest_path, self.manifest)
        print(f'{len(self.y):,} train, {len(self.test):,} test, {n_folds} folds | {self.output}', flush=True)

    def model_params(self, experiment, matrices, reference):
        params = experiment.params.copy()
        if experiment.family == 'glm':
            n_numeric = len(self.config['numeric_columns'])
            n_flags = len(self.config['flag_columns'])
            first_log = n_numeric + n_flags
            categories = self.train.iloc[reference][self.config['categorical_columns']]
            raw_width = first_log + 2 + int(categories.nunique(dropna=False).sum())
            columns = [*range(n_numeric), first_log, first_log + 1]
            columns.extend(range(raw_width, matrices[0].shape[1]))
            params['standardize_cols'] = columns
            if self.smoke:
                params['max_iter'] = min(30, params.get('max_iter', 30))
        elif self.smoke:
            params.update(max_iterations=24, patience=5)
        return params

    def matrices(self, experiment, partition, queries):
        return self.builder.make(
            partition, queries, groups=experiment.groups, smoothing=experiment.smoothing,
            seed=experiment.encoder_seed, logit_te=experiment.family == 'glm',
        )

    def fit_or_refit(self, experiment, reference, queries, *, early=None, rounds=None):
        from models import fit_predict

        all_queries = (early,) if early is not None else queries
        matrices = self.matrices(experiment, reference, all_queries)
        fit, *pred = matrices
        params = self.model_params(experiment, matrices, reference)
        offsets = None
        if experiment.boost:
            glm_spec = replace(experiment, family='glm', groups=('glm_raw', *experiment.groups[1:]), boost=False,
                               params={'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 0.0001})
            glm_matrices = self.matrices(glm_spec, reference, all_queries)
            glm_fit, *glm_queries = glm_matrices
            glm_predictions, _ = fit_predict(
                'glm', glm_fit, self.y[reference], (glm_fit, *glm_queries),
                params=self.model_params(glm_spec, glm_matrices, reference), seed=experiment.model_seed,
            )
            train_margin, *query_margin = [logit(np.clip(p.astype(float), 1e-7, 1 - 1e-7)) for p in glm_predictions]
            offsets = (train_margin, query_margin[0] if early is not None else None,
                       () if early is not None else tuple(query_margin))
        common = dict(params=params, seed=experiment.model_seed, rounds=rounds, offsets=offsets)
        if early is not None:
            return fit_predict(experiment.family, fit, self.y[reference], (),
                               early=(pred[0], self.y[early]), **common)
        return fit_predict(experiment.family, fit, self.y[reference], tuple(pred), **common)

    def fit_fold(self, experiment, part, queries):
        start = perf_counter()
        rounds = None
        early_details = None
        if experiment.family != 'glm':
            _, early_details = self.fit_or_refit(experiment, part['A'], (), early=part['E'])
            rounds = max(1, round(early_details['rounds'] * experiment.rounds_factor))
        predictions, details = self.fit_or_refit(experiment, part['T'], queries, rounds=rounds)
        for prediction, indexes in zip(predictions, queries, strict=True):
            if (prediction.shape != (len(indexes),) or not np.isfinite(prediction).all()
                    or np.any(prediction < 0) or np.any(prediction > 1)):
                raise ValueError(f'Invalid {experiment.name} predictions')
        if 'RTX 4060' not in details.get('device', '') or (
                early_details and 'RTX 4060' not in early_details.get('device', '')):
            raise RuntimeError(f'GPU device could not be verified: {details} / {early_details}')
        details['early'] = early_details
        details['seconds'] = perf_counter() - start
        return predictions, details

    def run(self, experiment):
        directory = self.output / experiment.name
        directory.mkdir(exist_ok=True)
        spec_path = directory / 'config.json'
        specification = json.loads(json.dumps(asdict(experiment)))
        if spec_path.exists() and read_json(spec_path) != specification:
            raise ValueError(f'The saved config changed for {experiment.name}')
        write_json(spec_path, specification)
        oof = np.full(len(self.y), np.nan, dtype=np.float64)
        test_by_fold = np.empty((len(self.test), len(self.partitions)), dtype=np.float64)
        fold_details = []
        for number, part in enumerate(self.partitions):
            valid = part['V']
            data_path = directory / f'fold_{number + 1}.npz'
            metric_path = directory / f'fold_{number + 1}.json'
            if data_path.exists() and metric_path.exists():
                with np.load(data_path) as saved:
                    if not np.array_equal(saved['ids'], self.ids[valid]):
                        raise ValueError(f'Fold IDs changed for {experiment.name}')
                    pred, test_pred = saved['valid'], saved['test']
                details = read_json(metric_path)
                print(f'{experiment.name} fold {number + 1}: resumed AUC {details["auc"]:.6f}', flush=True)
            else:
                print(f'{experiment.name} fold {number + 1}: running on GPU', flush=True)
                (pred, test_pred), details = self.fit_fold(experiment, part, (valid, self.test_indices))
                details.update(fold=number + 1, auc=float(roc_auc_score(self.y[valid], pred)),
                               train_rows=len(part['T']), valid_rows=len(valid))
                write_npz(data_path, ids=self.ids[valid], valid=pred, test=test_pred)
                write_json(metric_path, details)
                print(f'{experiment.name} fold {number + 1}: AUC {details["auc"]:.6f} | '
                      f'{details["seconds"]:.1f}s | {details["device"]}', flush=True)
            oof[valid] = pred
            test_by_fold[:, number] = test_pred
            fold_details.append(details)
        metrics = self.save_predictions(experiment.name, oof, test_by_fold.mean(axis=1),
                                        specification, fold_details, test_by_fold=test_by_fold)
        return metrics

    def load(self, name):
        with np.load(self.output / name / 'predictions.npz') as saved:
            if not np.array_equal(saved['ids'], self.ids) or not np.array_equal(saved['test_ids'], self.test_ids):
                raise ValueError(f'Prediction IDs mismatch for {name}')
            return saved['oof'], saved['test']

    def save_predictions(self, name, oof, test, config, folds=None, **extra):
        directory = self.output / name
        directory.mkdir(exist_ok=True)
        if not np.isfinite(oof).all() or not np.isfinite(test).all():
            raise ValueError(f'Incomplete predictions for {name}')
        folds = folds or [{'fold': int(fold + 1), 'auc': float(roc_auc_score(self.y[self.folds == fold],
                                                                             oof[self.folds == fold]))}
                          for fold in np.unique(self.folds)]
        fold_scores = [row['auc'] for row in folds]
        metrics = {
            'name': name, 'status': 'completed', 'config': config, **self.manifest,
            **metric_row(self.y, oof), 'fold_mean': float(np.mean(fold_scores)),
            'fold_std': float(np.std(fold_scores)), 'folds': folds,
            'seconds': sum(fold.get('seconds', 0.0) for fold in folds),
        }
        parent = config.get('parent')
        if parent and (parent_path := self.output / parent / 'metrics.json').is_file():
            baseline = read_json(parent_path)
            metrics['delta_oof_auc'] = metrics['oof_auc'] - baseline['oof_auc']
            metrics['delta_by_fold'] = [score - row['auc'] for score, row in zip(fold_scores, baseline['folds'])]
        write_npz(directory / 'predictions.npz', ids=self.ids, test_ids=self.test_ids,
                  fold=self.folds, oof=oof, test=test, **extra)
        write_json(directory / 'metrics.json', metrics)
        self.reports[name] = metrics
        self.write_comparison()
        print(f'{name} complete: OOF AUC {metrics["oof_auc"]:.6f}', flush=True)
        return metrics

    def write_comparison(self):
        rows = []
        for path in sorted(self.output.glob('*/metrics.json')):
            result = read_json(path)
            if result.get('status') != 'completed':
                continue
            row = {name: result.get(name) for name in (
                'name', 'oof_auc', 'delta_oof_auc', 'average_precision', 'fold_mean', 'fold_std', 'log_loss', 'seconds',
            )}
            row['parent'] = result['config'].get('parent', '')
            for fold in result['folds']:
                row[f'fold_{fold["fold"]}'] = fold['auc']
            rows.append(row)
        if rows:
            path = self.output / 'comparison.csv'
            temp = path.with_suffix(f'.csv.{os.getpid()}.tmp')
            pd.DataFrame(rows).sort_values('oof_auc', ascending=False).to_csv(temp, index=False, float_format='%.10f')
            temp.replace(path)

    def score(self, experiment):
        name = experiment if isinstance(experiment, str) else experiment.name
        return read_json(self.output / name / 'metrics.json')['oof_auc']

    def best(self, experiments):
        return max(experiments, key=self.score)

    def blend(self, name, members, rank=False, weights=None, parent=''):
        weights = list(weights) if weights is not None else [1 / len(members)] * len(members)
        matrices = list(zip(*(self.load(member) for member in members), strict=True))
        scores = [np.average([rankdata(col) / len(col) if rank else col for col in group],
                             weights=weights, axis=0) for group in matrices]
        return self.save_predictions(name, *scores, {'members': members, 'weights': weights,
                                                     'parent': parent, 'rank': rank})

    def run_te(self, requested=None):
        pairs = get_specs()
        for base, experiment in pairs:
            needs_prerequisite = (requested is not None and 'T04' in requested
                                  and not (self.output / experiment.name / 'metrics.json').is_file())
            if requested is None or experiment.name in requested or needs_prerequisite:
                self.run(base)
                self.run(experiment)
        versions = [spec for _, spec in pairs]
        if requested is None or 'T04' in requested:
            best = self.best(versions)
            te24 = replace(best, name='T04', groups=(best.groups[0], 'te24'), parent=best.name)
            self.run(te24)
            versions.append(te24)
        return versions

    def run_features(self, versions):
        anchor = self.best(versions)
        glm = next(spec for spec in versions if spec.name == 'T01')
        specs = [
            replace(anchor, name='D01', groups=(*anchor.groups, 'cents'), parent=anchor.name),
            replace(anchor, name='D02', groups=(*anchor.groups, 'digits'), parent=anchor.name),
            replace(anchor, name='D03', groups=(*anchor.groups, 'cents', 'digits'), parent=anchor.name),
            replace(glm, name='P1', groups=(*glm.groups, 'p1'), parent=glm.name),
            replace(glm, name='P1-P2', groups=(*glm.groups, 'p1', 'p2'), parent='P1'),
            replace(glm, name='P1-P2-P3', groups=(*glm.groups, 'p1', 'p2', 'p3'), parent='P1-P2'),
            replace(anchor, name='COUNT', groups=(*anchor.groups, 'count'), parent=anchor.name),
            replace(anchor, name='DERIVED', groups=(*anchor.groups, 'derived'), parent=anchor.name),
            replace(glm, name='T01-M50', smoothing=50, parent=glm.name),
        ]
        for experiment in specs:
            self.run(experiment)
        return [*versions, *specs]

    def run_xgb_digit_comparison(self, versions):
        """Repeat the digit ablation with the matched XGBoost TE13 control."""
        anchor = next(spec for spec in versions if spec.name == 'T03')
        variants = [
            replace(anchor, name=f'XGB-{name}', groups=(*anchor.groups, *groups), parent=anchor.name)
            for name, groups in (('D01', ('cents',)), ('D02', ('digits',)), ('D03', ('cents', 'digits')))
        ]
        for spec in variants:
            self.run(spec)
        return [*versions, *variants]

    def run_advanced(self, versions, tuning_family=None, seed_anchor=None):
        glm = self.best([spec for spec in versions if spec.family == 'glm'])
        tree = self.best([spec for spec in versions if spec.family != 'glm'
                          and (tuning_family is None or spec.family == tuning_family)])
        cases = []
        for value in (0.1, 0.3):
            cases.append(replace(glm, name=f'GLM-C{value}', parent=glm.name, params={**glm.params, 'C': value}))
        cases.append(replace(glm, name='GLM-NEWTON', parent=glm.name,
                             params={**glm.params, 'solver': 'newton-cholesky'}))
        for variant, change in (
            ('LAMBDA3', {'reg_lambda': 3.0}), ('ETA005', {'learning_rate': 0.05}),
            ('NOSUB', {'subsample': 1.0, 'colsample_bytree': 1.0}),
        ):
            cases.append(replace(tree, name=f'TUNE-{variant}', parent=tree.name, params={**tree.params, **change}))
        tree_variants = [spec for spec in versions if spec.name in ('T02', 'T03')]
        for spec in tree_variants:
            cases.append(replace(spec, name=f'GLM-BOOST-{spec.family.upper()}', parent=spec.name, boost=True))
        for experiment in cases:
            self.run(experiment)
        candidates = [*versions, *cases]
        best = self.best(candidates) if seed_anchor is None else next(spec for spec in candidates if spec.name == seed_anchor)
        seed_specs = seed_variants(best)
        for spec in seed_specs:
            self.run(spec)
        members = [best.name, *(item.name for item in seed_specs)]
        self.blend('SEED3-MEAN', members, parent=best.name)
        self.blend('SEED3-RANK', members, rank=True, parent=best.name)
        return candidates, best, seed_specs

    def make_partition(self, train, valid):
        fit, early = train_test_split(train, test_size=0.1, random_state=self.config['early_stop_seed'],
                                      stratify=self.y[train])
        return {'T': np.sort(train), 'A': np.sort(fit), 'E': np.sort(early), 'V': valid}


def seed_variants(anchor):
    return [replace(anchor, name=f'{anchor.name}-S71', parent=anchor.name, model_seed=71),
            replace(anchor, name=f'{anchor.name}-S93', parent=anchor.name, encoder_seed=2028)]


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['te', 'features', 'advanced', 'all'], default='all')
    parser.add_argument('--only-te', nargs='+', choices=['T01', 'T02', 'T03', 'T04'])
    parser.add_argument('--compare-xgb-digits', action='store_true')
    parser.add_argument('--tune-family', choices=['xgb', 'lgb'])
    parser.add_argument('--seed-anchor')
    parser.add_argument('--stack-lgb')
    parser.add_argument('--stack-xgb')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE / 'results' / ('smoke' if args.smoke else 'full')
    run = PlanRun(output, smoke=args.smoke)
    if args.only_te and args.stage != 'te':
        parser.error('--only-te is available with --stage te')
    models = run.run_te(args.only_te)
    if args.stage == 'te':
        return
    models = run.run_features(models)
    if args.compare_xgb_digits:
        models = run.run_xgb_digit_comparison(models)
    if args.stage == 'features':
        return
    candidates, best, seeds = run.run_advanced(models, tuning_family=args.tune_family, seed_anchor=args.seed_anchor)
    if args.stage == 'all':
        from finalize import finalize

        finalize(run, candidates, best, seeds, stack_lgb=args.stack_lgb, stack_xgb=args.stack_xgb)


if __name__ == '__main__':
    cli()
