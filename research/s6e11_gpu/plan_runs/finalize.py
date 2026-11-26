"""Confirm the development candidates and create a validated Kaggle CSV locally."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256 as digest
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ensemble import bootstrap_delta, error_groups, evaluate_ensembles, fit_meta_fold, validate_submission
from run_plan import Experiment, PlanRun, TARGET, get_specs, metric_row, read_json, sha256, write_json, write_npz

INNER_VALIDATION_SEED = 2026


def groups_for_errors(run):
    train = run.train
    no_flags = train[run.config['flag_columns']].sum(axis=1) == 0
    amount_count = train.groupby('amount_usd')['id'].transform('size')
    return {
        'no_flags': no_flags,
        'auth_method': train['auth_method'],
        'cvv_retry_count': train['cvv_retry_count'],
        'merchant_category': train['merchant_category'],
        'age_band': pd.cut(train['customer_age'], [0, 25, 45, 65, np.inf], right=False).astype(str),
        'rare_amount': amount_count <= 5,
    }


def best_stack(metrics):
    return max((item for item in metrics if item['method'] in ('logistic_stack', 'rank_logit_stack')),
               key=lambda item: item['auc'])


def choose_nested_stack(metrics, specs):
    full_stack = best_stack(metrics)
    alternatives = [item for item in metrics if item['method'] == 'leave_one_out'
                    and item['full_stack'] == full_stack['candidate']
                    and item['passes_practical_screen'] and full_stack['auc'] - item['auc'] < 0.0001]
    if not alternatives:
        return full_stack, specs
    simpler = max(alternatives, key=lambda item: item['auc'])
    members = [spec for spec in specs if spec.name != simpler['omitted']]
    return {**simpler, 'method': full_stack['method']}, members


def nested_inner_oof(run, spec, outer_fold, seed, output):
    outer_train = run.partitions[outer_fold - 1]['T']
    scores = np.full(len(outer_train), np.nan)
    spec_hash = digest(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()
    splitter = StratifiedKFold(n_splits=len(run.partitions), shuffle=True, random_state=seed)
    for inner_fold, (fit, valid) in enumerate(splitter.split(outer_train, run.y[outer_train]), start=1):
        path = output / f'outer_{outer_fold}_inner_{inner_fold}_{spec.name}.npz'
        valid_ids = run.ids[outer_train[valid]]
        if path.exists():
            with np.load(path) as saved:
                if (not np.array_equal(saved['ids'], valid_ids) or saved['seed'].item() != seed
                        or saved['spec_hash'].item() != spec_hash):
                    raise ValueError('Nested inner inputs changed')
                scores[valid] = saved['prediction']
            continue
        partition = run.make_partition(outer_train[fit], outer_train[valid])
        print(f'Nested outer {outer_fold} inner {inner_fold} | {spec.name}', flush=True)
        (prediction,), meta = run.fit_fold(spec, partition, (partition['V'],))
        scores[valid] = prediction
        write_npz(path, ids=valid_ids, prediction=prediction, rounds=meta['rounds'], device=meta['device'],
                  seed=seed, spec_hash=spec_hash)
    return scores


def populate_nested_parallel(run, specs, seed, output):
    folds = range(1, len(run.partitions) + 1)
    pending = [fold for fold in folds if not (output / f'outer_{fold}.npz').is_file()]
    if not pending or run.smoke:
        return
    jobs = []
    for spec in specs:
        # LightGBM's OpenCL fit leaves GPU capacity for another fold batch.
        groups = [pending[::2], pending[1::2]] if spec.family == 'lgb' else [pending]
        for outer_folds in filter(None, groups):
            payload = {
                'spec': asdict(spec), 'outer_folds': outer_folds,
                'seed': seed, 'results': str(run.output), 'directory': str(output),
            }
            job = output / f'worker_{spec.name}_{outer_folds[0]}.json'
            write_json(job, payload)
            process = subprocess.Popen([sys.executable, '-u', __file__, str(job)])
            jobs.append((job, process))
    for job, process in jobs:
        if process.wait() != 0:
            print(f'{job.stem} stopped early; resume its pending folds in the main run', flush=True)


def run_nested_worker(job):
    value = read_json(Path(job))
    run = PlanRun(Path(value['results']))
    data = value['spec']
    data['groups'] = tuple(data['groups'])
    spec = Experiment(**data)
    for outer_fold in value['outer_folds']:
        nested_inner_oof(run, spec, outer_fold, value['seed'], Path(value['directory']))


def validate_nested(run, specs, stack, oof_arrays, output):
    """Refit inner base models using only the current outer training split."""
    output.mkdir(exist_ok=True)
    stack_choice = {key: stack[key] for key in ('candidate', 'c', 'method', 'meta_seed')}
    configuration = {
        'members': [spec.name for spec in specs], 'stack': stack_choice,
        'base_prediction_hashes': [digest(scores.tobytes()).hexdigest() for scores in oof_arrays],
        'outer_folds': len(run.partitions), 'inner_folds': len(run.partitions), 'inner_seed': INNER_VALIDATION_SEED,
        'fold_hash': run.fold_hash, 'smoke': run.smoke,
    }
    config_path = output / 'config.json'
    if config_path.exists() and read_json(config_path) != configuration:
        raise ValueError('Nested validation selection changed since checkpoint')
    write_json(config_path, configuration)
    populate_nested_parallel(run, specs, INNER_VALIDATION_SEED, output)
    stack_oof = np.full(len(run.y), np.nan)
    score_rows = []
    controls = np.column_stack(oof_arrays)
    for fold, part in enumerate(run.partitions, start=1):
        outer_train, outer_valid = part['T'], part['V']
        score_path = output / f'outer_{fold}.npz'
        if score_path.exists():
            with np.load(score_path) as cached:
                if not np.array_equal(cached['ids'], run.ids[outer_valid]):
                    raise ValueError('Nested checkpoint IDs changed')
                prediction = cached['prediction']
        else:
            inner_oof = np.column_stack([
                nested_inner_oof(run, spec, fold, INNER_VALIDATION_SEED, output) for spec in specs
            ])
            if not np.isfinite(inner_oof).all():
                raise ValueError('Nested meta features are incomplete')
            (prediction,) = fit_meta_fold(inner_oof, run.y[outer_train], controls[outer_valid],
                                          c=stack['c'], rank_logit=stack['method'] == 'rank_logit_stack')
            write_npz(score_path, ids=run.ids[outer_valid], prediction=prediction)
        stack_oof[outer_valid] = prediction
        score = float(roc_auc_score(run.y[outer_valid], prediction))
        score_rows.append(score)
        print(f'Nested outer {fold}: AUC {score:.6f}', flush=True)
    write_npz(output / 'predictions.npz', ids=run.ids, fold=run.folds, oof=stack_oof)
    evaluation = {**configuration, **metric_row(run.y, stack_oof), 'fold_auc': score_rows,
                  'selection_context': 'candidate families and meta hyperparameters chosen in development CV'}
    write_json(output / 'metrics.json', evaluation)
    return stack_oof, evaluation


def candidate_comparison(run, prediction, control, repeats):
    delta = float(roc_auc_score(run.y, prediction) - roc_auc_score(run.y, control))
    delta_by_fold = []
    for fold in np.unique(run.folds):
        keep = run.folds == fold
        delta_by_fold.append(float(roc_auc_score(run.y[keep], prediction[keep])
                                   - roc_auc_score(run.y[keep], control[keep])))
    supported = delta >= 0.0001 and sum(item > 0 for item in delta_by_fold) >= len(delta_by_fold) - 1
    return {'delta': delta, 'delta_by_fold': delta_by_fold, 'passes_screen': supported,
            'bootstrap': bootstrap_delta(run.y, prediction, control, repeats=repeats)}


def evaluate_refit(run, spec, output):
    from dataclasses import replace

    output.mkdir(exist_ok=True)
    path = output / 'comparison.json'
    if path.is_file():
        previous = read_json(path)
        if previous['model'] != spec.name:
            raise ValueError('Full refit checkpoint belongs to a different model')
        return previous
    # Match the existing outer fitting protocol for the 1.0 control. Only the
    # number of rounds changes in the 1.25 variant. Both use the same CV split.
    name = f'{spec.name}-R125'
    base = read_json(run.output / spec.name / 'metrics.json')
    candidate = replace(spec, name=name, parent=spec.name, rounds_factor=1.25)
    result = run.run(candidate)
    source = result if result['oof_auc'] > base['oof_auc'] else base
    selected = candidate if source is result else spec
    # Short smoke and full artifact use the same refit implementation.
    count = int(np.median([fold['rounds'] for fold in source['folds']]))
    indices = np.arange(len(run.y), dtype=np.int64)
    (refit,), details = run.fit_or_refit(selected, indices, (run.test_indices,), rounds=count)
    _, average = run.load(spec.name)
    if (refit.shape != average.shape or not np.isfinite(refit).all()
            or np.any(refit < 0) or np.any(refit > 1)):
        raise ValueError('Full-train refit returned invalid scores')
    write_npz(output / 'test_refit.npz', id=run.test_ids, full=refit, fold_mean=average)
    correlation = float(spearmanr(refit, average).statistic)
    comparison = {'model': spec.name, 'selected': selected.name, 'rounds': count,
                  'base_oof_auc': base['oof_auc'], 'r125_oof_auc': result['oof_auc'],
                  'test_mean_abs_delta': float(np.abs(refit - average).mean()),
                  'test_spearman': correlation if np.isfinite(correlation) else None, 'device': details['device'],
                  'mode': 'alternate full-data fit; final path keeps fold average'}
    write_json(path, comparison)
    return comparison


def replay_final_test(run, output, selection, specs, nested_specs, best, seed_anchor, seeds, meta_train, stack):
    from bundle import export_members

    if selection.startswith('SEED3-'):
        members = [seed_anchor, *seeds]
    elif selection.startswith('TREES2-'):
        members = specs[1:]
    elif selection == 'NESTED-STACK':
        members = nested_specs
    elif selection.startswith('ALL3-'):
        members = specs
    else:
        members = [best]
    test_arrays, manifest = export_members(run, members, output / 'model_bundle')
    values = [test_arrays[spec.name] for spec in members]
    if selection == 'NESTED-STACK':
        (score,), _ = fit_meta_fold(meta_train, run.y, np.column_stack(values), c=stack['c'],
                                   rank_logit=stack['method'] == 'rank_logit_stack', return_details=True)
    elif selection.endswith('-RANK'):
        score = np.mean([rankdata(value) / len(value) for value in values], axis=0)
    else:
        score = np.mean(values, axis=0)
    return score, manifest


def finalize(run, candidates, best, seeds, stack_lgb=None, stack_xgb=None):
    repeats = 20 if run.smoke else 500
    groups = groups_for_errors(run)
    controls = [control for control, _ in get_specs()]
    all_bases = [*candidates, *controls]
    seed_anchor = best
    best = run.best(all_bases)
    fixed_members = {'lgb': stack_lgb, 'xgb': stack_xgb}
    specs = [run.best([candidate for candidate in all_bases if candidate.family == family
                       and (not fixed_members.get(family) or candidate.name == fixed_members[family])])
             for family in ('glm', 'lgb', 'xgb')]
    names = [spec.name for spec in specs]
    pairs = [run.load(name) for name in names]
    train_scores, test_scores = (dict(zip(names, values, strict=True)) for values in zip(*pairs, strict=True))
    development = run.output / 'ensemble'
    development.mkdir(exist_ok=True)
    info = development / 'ensemble_summary.json'
    inputs_path = development / 'inputs.json'
    sources = {name: sha256(run.output / name / 'predictions.npz') for name in names}
    if info.exists():
        if not inputs_path.is_file() or read_json(inputs_path) != sources:
            raise ValueError('Ensemble checkpoint belongs to a different set of predictions')
        results = read_json(info)
    else:
        write_json(inputs_path, sources)
        results = evaluate_ensembles(run.y, train_scores, test_scores, run.folds, output=development,
                                     rank_logit=True, repeats=repeats, ids=run.ids, groups=groups)
    stack, nested_specs = choose_nested_stack(results['metrics'], specs)
    nested_base_oof = [train_scores[spec.name] for spec in nested_specs]
    nested_pred, nested = validate_nested(run, nested_specs, stack, nested_base_oof, run.output / 'nested')
    meta_train = np.column_stack(nested_base_oof)
    meta_test = np.column_stack([test_scores[spec.name] for spec in nested_specs])
    (stack_test,), stack_fit = fit_meta_fold(meta_train, run.y, meta_test, c=stack['c'],
                                           rank_logit=stack['method'] == 'rank_logit_stack', return_details=True)
    selected_name = best.name
    selected_oof, selected_test = run.load(selected_name)
    reference = selected_oof
    best_reference = best.name
    alternatives = {'NESTED-STACK': (nested_pred, stack_test)}
    trees = names[1:]
    for tag, rank in (('MEAN', False), ('RANK', True)):
        for scope, members in (('ALL3', names), ('TREES2', trees)):
            label = f'{scope}-{tag}'
            run.blend(label, members, rank=rank, parent=best_reference)
            alternatives[label] = run.load(label)
        label = f'SEED3-{tag}'
        alternatives[label] = run.load(label)
    evidence = {name: candidate_comparison(run, scores, reference, repeats)
                for name, (scores, _) in alternatives.items()}
    screened = [name for name, record in evidence.items() if record['passes_screen']]
    if screened:
        selected_name = max(screened, key=lambda name: evidence[name]['delta'])
        selected_oof, selected_test = alternatives[selected_name]
    best_tree = run.best(specs[1:])
    refit = evaluate_refit(run, best_tree, run.output / 'full_refit')
    final = run.output / 'final'
    final.mkdir(exist_ok=True)
    score_kind = 'fractional_rank' if selected_name.endswith('-RANK') else 'model_probability'
    run_id = f'{run.output.name}-{run.fold_hash[:12]}'
    test_from_bundle, bundle = replay_final_test(run, final, selected_name, specs, nested_specs, best,
                                               seed_anchor, seeds, meta_train, stack)
    test_delta = float(np.max(np.abs(test_from_bundle - selected_test)))
    selected_test = test_from_bundle
    run.save_predictions('FINAL', selected_oof, selected_test, {
        'parent': best_reference, 'selected': selected_name, 'score_kind': score_kind,
    })
    error_groups(run.y, run.ids, groups, [(best_reference, reference), (selected_name, selected_oof)]).to_csv(
        final / 'error_groups.csv', index=False,
    )
    selection = {
        'run_id': run_id, 'smoke': run.smoke, 'selected': selected_name, 'score_kind': score_kind,
        'base_specs': [asdict(spec) for spec in specs],
        'seed_specs': [asdict(spec) for spec in [seed_anchor, *seeds]],
        'best_single': best_reference, 'oof_auc': float(roc_auc_score(run.y, selected_oof)),
        'nested': nested, 'evidence': evidence, 'full_train_alternate': refit, 'fold_hash': run.fold_hash,
        'stack_fit_on_all_base_oof': stack_fit if selected_name == 'NESTED-STACK' else None,
        'model_bundle': {'members': bundle['members'], 'path': str(final / 'model_bundle'),
                         'max_delta_vs_original_test_prediction': test_delta},
    }
    if not run.smoke:
        path = final / 'submission.csv'
        temp = path.with_suffix('.csv.tmp')
        pd.DataFrame({'id': run.test_ids, TARGET: selected_test}).to_csv(temp, index=False)
        selection['submission'] = validate_submission(temp, run.sample, run_id=run_id, score_kind=score_kind)
        temp.replace(path)
        selection['submission']['path'] = str(path)
        if sha256(path) != selection['submission']['sha256']:
            raise ValueError('Submission changed after validation')
    write_json(final / 'selection.json', selection)
    print(f'Selected {selected_name} | OOF AUC {selection["oof_auc"]:.6f} | {final}', flush=True)


if __name__ == '__main__':
    run_nested_worker(sys.argv[1])
