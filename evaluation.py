"""
Evaluation harness: measures the QA pipeline against its real goal.

Proxies such as normality (transform choice) or SHAP importance (feature choice) are useful
for exploration, but the question that matters is: does the pipeline catch bad data without
burying analysts in false alarms? This module answers it by planting realistic errors of
known types and counting what the full validator (schema + model) catches.

Protocol (no information leaks from test into any decision):
  1. The model, transform and threshold are fitted on the TRAIN split only.
  2. Candidate configurations are compared on the VALIDATION split, over several random
     seeds of planted errors, and one is chosen by the rule in choose_config().
  3. The chosen configuration is scored ONCE on the TEST split. That number is the
     honest estimate of performance on unseen countries.

Run after data_prep.py:  python evaluation.py
"""
import contextlib
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from transforms import PARAMETER_FREE
from validation import EmissionsDataValidator

ERROR_TYPES = ['unit_x1000', 'unit_div1000', 'decimal_shift_x10', 'sign_flip',
               'column_swap', 'duplicate_record', 'missing_value']
# Errors that no single-column rule can catch: every value stays individually legal,
# so only the statistical layer can find them. The model is chosen on these.
STATISTICAL_TYPES = ['unit_x1000', 'unit_div1000', 'decimal_shift_x10', 'column_swap']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _features(df):
    return [c for c in df.select_dtypes(include='number').columns if c != 'year']


def inject_errors(df, features, n_per_type=20, split='validation', seed=0):
    """
    Plants `n_per_type` errors of every type into distinct rows of `split`.
    Returns (corrupted copy, labels) where labels has one row per planted error.
    """
    rng = np.random.default_rng(seed)
    bad = df.reset_index(drop=True).copy()
    for col in features:
        bad[col] = bad[col].astype(float)
    pool = bad.index[bad['split'] == split].to_numpy()
    rows = rng.choice(pool, size=n_per_type * len(ERROR_TYPES), replace=False)

    labels, copies = [], []
    for k, error_type in enumerate(ERROR_TYPES):
        for row in rows[k * n_per_type:(k + 1) * n_per_type]:
            row = int(row)
            if error_type == 'column_swap':
                a, b = (str(c) for c in rng.choice(features, size=2, replace=False))
                bad.loc[row, [a, b]] = bad.loc[row, [b, a]].to_numpy()
                labels.append({'row': row, 'error_type': error_type, 'column': f'{a}<->{b}'})
                continue
            if error_type == 'duplicate_record':
                copies.append(bad.loc[[row]])
                labels.append({'row': len(bad) + len(copies) - 1, 'error_type': error_type,
                               'column': 'country, year', 'original_row': row})
                continue
            positive = [c for c in features if bad.at[row, c] > 0]
            col = str(rng.choice(positive if error_type == 'sign_flip' and positive else features))
            value = bad.at[row, col]
            bad.at[row, col] = {
                'unit_x1000': value * 1000,
                'unit_div1000': value / 1000,
                'decimal_shift_x10': value * 10,
                'sign_flip': -abs(value),
                'missing_value': np.nan,
            }[error_type]
            labels.append({'row': row, 'error_type': error_type, 'column': col})

    if copies:
        bad = pd.concat([bad] + copies, ignore_index=True)
    return bad, pd.DataFrame(labels)


def evaluate_config(df, config, n_per_type=20, split='validation', seed=0):
    """Plants errors into `split`, runs the full validator, and scores what it caught."""
    bad, labels = inject_errors(df, _features(df), n_per_type=n_per_type, split=split, seed=seed)
    validator = EmissionsDataValidator(bad, explain=False, **config)
    with contextlib.redirect_stdout(io.StringIO()):
        validator.run()
    by_schema = validator.invalid_idx
    by_model = set(validator.scored.index[validator.scored['is_anomaly']])
    flagged = by_schema | by_model

    labels = labels.assign(by_schema=labels['row'].isin(by_schema),
                           by_model=labels['row'].isin(by_model))
    labels['caught'] = labels['by_schema'] | labels['by_model']
    per_type = (labels.groupby('error_type')
                .agg(injected=('row', 'size'), caught=('caught', 'sum'),
                     by_schema=('by_schema', 'sum'), by_model=('by_model', 'sum'))
                .reindex(ERROR_TYPES).reset_index())
    per_type['recall'] = per_type['caught'] / per_type['injected']

    # Clean rows = rows of the split with nothing planted (the original of a duplicate is
    # excluded too: the uniqueness rule rightly reports both copies)
    originals = set(labels.get('original_row', pd.Series(dtype=float)).dropna().astype(int))
    split_rows = set(bad.index[bad['split'] == split])
    clean = split_rows - set(labels['row']) - originals
    false_pos = len(clean & flagged)
    true_pos = int(labels['caught'].sum())
    summary = {
        'recall_statistical': per_type.set_index('error_type').loc[STATISTICAL_TYPES, 'recall'].mean(),
        'recall_all': per_type['recall'].mean(),
        'false_positive_rate': false_pos / len(clean) if clean else 0.0,
        'precision': true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0,
        'clean_rows': len(clean),
    }
    return per_type, summary


def run_grid(df, grid, n_per_type=20, split='validation', seeds=(0, 1, 2)):
    """Evaluates every configuration over several seeds of planted errors; one averaged row each."""
    rows = []
    for config in grid:
        results = [evaluate_config(df, config, n_per_type, split, seed) for seed in seeds]
        summary = pd.DataFrame([s for _, s in results]).mean().to_dict()
        recalls = pd.concat([p.set_index('error_type')['recall'] for p, _ in results], axis=1).mean(axis=1)
        rows.append({**config, **summary, **{f'recall_{t}': recalls[t] for t in ERROR_TYPES}})
        print(f"  {config} -> statistical recall {summary['recall_statistical']:.3f}, "
              f"false-positive rate {summary['false_positive_rate']:.3f}")
    return pd.DataFrame(rows)


def choose_config(results, fpr_budget=0.03):
    """
    Highest statistical recall among configurations whose false-positive rate fits the
    review budget (the share of clean records an analyst can afford to check by hand).
    If none fits, the configuration with the lowest false-positive rate.
    """
    within = results[results['false_positive_rate'] <= fpr_budget]
    if len(within):
        best = within.sort_values(['recall_statistical', 'false_positive_rate'], ascending=[False, True]).iloc[0]
        reason = (f"Highest recall on errors the schema cannot catch ({best['recall_statistical']:.1%}) among "
                  f"configurations within the {fpr_budget:.0%} false-positive budget ({fpr_budget}); "
                  f"validation false-positive rate {best['false_positive_rate']:.1%}.")
    else:
        best = results.sort_values('false_positive_rate').iloc[0]
        reason = (f"No configuration met the {fpr_budget:.0%} false-positive budget ({fpr_budget}); "
                  f"chose the lowest false-positive rate ({best['false_positive_rate']:.1%}).")
    config = {'transform': str(best['transform']), 'use_ratios': bool(best['use_ratios']),
              'contamination': float(best['contamination']),
              'range_z': None if pd.isna(best.get('range_z')) else float(best['range_z'])}
    return config, reason


def default_grid():
    transforms = PARAMETER_FREE + ['yeo_johnson']   # box_cox is excluded: it cannot take zero/negative values
    return [{'transform': t, 'use_ratios': r, 'contamination': c, 'range_z': z}
            for t in transforms for r in (True, False) for c in (0.01, 0.02, 0.03)
            for z in (None, 4.0, 6.0)]


def run_evaluation(data_path=os.path.join(BASE_DIR, 'data', 'shap_selected_emissions_data.csv'),
                   report_dir=os.path.join(BASE_DIR, 'reports'), fpr_budget=0.03, n_per_type=20,
                   seeds=(0, 1, 2), grid=None):
    df = pd.read_csv(data_path)
    if 'split' not in df:
        raise ValueError("Dataset has no 'split' column; run data_prep.py first.")
    os.makedirs(report_dir, exist_ok=True)

    print(f"INFO: Comparing configurations on the validation split ({len(seeds)} seeds x "
          f"{n_per_type} planted errors per type)...")
    results = run_grid(df, grid or default_grid(), n_per_type, 'validation', seeds)
    results.to_csv(f"{report_dir}/evaluation_grid.csv", index=False)
    config, reason = choose_config(results, fpr_budget)
    print(f"\nINFO: Chosen configuration: {config}. {reason}")

    # The test split is touched exactly once, with the configuration already fixed
    reports = {}
    for split in ('validation', 'test'):
        runs = [evaluate_config(df, config, n_per_type, split, seed) for seed in seeds]
        per_type = pd.concat([p for p, _ in runs]).groupby('error_type', sort=False).mean(numeric_only=True).reset_index()
        per_type.insert(0, 'split', split)
        per_type.to_csv(f"{report_dir}/evaluation_{split}.csv", index=False)
        reports[split] = {k: round(float(v), 4) for k, v in pd.DataFrame([s for _, s in runs]).mean().items()}
        print(f"\n{split.upper()} (chosen configuration):")
        print(per_type[['error_type', 'recall', 'by_schema', 'by_model']].round(3).to_string(index=False))
        print(f"  false-positive rate {reports[split]['false_positive_rate']:.3f}, "
              f"statistical recall {reports[split]['recall_statistical']:.3f}")

    with open(f"{report_dir}/model_config.json", 'w', encoding='utf-8') as f:
        json.dump({**config, 'fpr_budget': fpr_budget, 'reason': reason,
                   'validation': reports['validation'], 'test': reports['test'],
                   'n_per_type': n_per_type, 'seeds': list(seeds),
                   'evaluated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}, f, indent=2)
    print(f"\nINFO: Wrote {report_dir}/model_config.json")
    return config, reports


if __name__ == "__main__":
    run_evaluation()
