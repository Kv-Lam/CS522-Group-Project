"""Train a fixed baseline and compare three predictors on identical held-out rows."""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplconfig'))


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(fields+[k for r in rows for k in r])))
        writer.writeheader(); writer.writerows(rows)


def features(row):
    """Whitelist prediction-time features; never include labels, IDs or future states."""
    numeric = ['latitude', 'longitude', 'along_route_distance_m', 'straight_line_distance_m',
               'gps_age_seconds', 'scheduled_remaining_seconds']
    result = {k: float(row[k]) for k in numeric}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite feature')
    for k in ('route_id', 'direction_id', 'target_stop_id'):
        if not row.get(k):
            raise ValueError('Missing categorical feature')
        result[k] = str(row[k])
    local = datetime.fromisoformat(row['observation_local'])
    hour = local.hour+local.minute/60+local.second/3600
    result.update(hour_sin=math.sin(2*math.pi*hour/24), hour_cos=math.cos(2*math.pi*hour/24),
                  weekday=str(local.weekday()))
    return result


def load_rows(runs, quality='all'):
    accepted, excluded, seen = [], [], set()
    for run in sorted(map(Path, runs)):
        schedules = {r['arrival_event_id']: r for r in read_csv(run / 'punctuality_events.csv')}
        for source in read_csv(run / 'training_samples.csv'):
            row = dict(source, source_run=str(run.resolve()))
            try:
                if row['sample_id'] in seen:
                    raise ValueError('Duplicate sample across input runs')
                seen.add(row['sample_id'])
                if quality == 'strict' and row['label_confidence'] == 'low_crossing':
                    raise ValueError('Crossing label excluded by strict quality filter')
                schedule = schedules.get(row['arrival_event_id'], {})
                if schedule.get('schedule_match_status') != 'matched':
                    raise ValueError('No reliable static schedule match')
                obs = float(row['observation_timestamp'])
                row['scheduled_remaining_seconds'] = float(schedule['scheduled_arrival_timestamp'])-obs
                if row.get('official_eta_timestamp') in ('', None):
                    raise ValueError('Missing official ETA; excluded from all three predictors')
                row['official_remaining_seconds'] = float(row['official_eta_timestamp'])-obs
                for k in ('remaining_travel_time_seconds', 'remaining_time_lower_seconds', 'remaining_time_upper_seconds'):
                    row[k] = float(row[k])
                values = [row[k] for k in ('scheduled_remaining_seconds', 'official_remaining_seconds',
                           'remaining_travel_time_seconds', 'remaining_time_lower_seconds', 'remaining_time_upper_seconds')]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('Nonfinite comparison value')
                if not 0 < row['remaining_time_lower_seconds'] <= row['remaining_travel_time_seconds'] <= row['remaining_time_upper_seconds']:
                    raise ValueError('Invalid or overlapping label interval')
                if row['service_date'] != row['trip_instance_id'].split('|')[1]:
                    raise ValueError('Service date mismatch')
                features(row)
                accepted.append(row)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                excluded.append(dict(sample_id=row.get('sample_id'), source_run=str(run), reason=str(exc)))
    return accepted, excluded


def split_rows(rows, mode='auto'):
    """Prefer a chronological date holdout; fall back to explicitly exploratory trip CV."""
    dates = sorted({r['service_date'] for r in rows})
    if mode == 'date' or (mode == 'auto' and len(dates) >= 2):
        if len(dates) < 2:
            return [], 'Insufficient dates for chronological holdout'
        count = max(1, math.ceil(len(dates)*.2))
        test_dates = set(dates[-count:])
        test = [i for i,r in enumerate(rows) if r['service_date'] in test_dates]
        cutoff = min(float(rows[i]['observation_timestamp']) for i in test)
        train = [i for i,r in enumerate(rows) if r['service_date'] not in test_dates and
                 float(r['observation_timestamp'])+r['remaining_time_upper_seconds'] < cutoff]
        return [(np.array(train, dtype=int), np.array(test, dtype=int))], 'Chronological service-date holdout; overlapping label intervals purged'
    groups = [r['trip_instance_id'] for r in rows]
    if len(set(groups)) < 3:
        return [], 'Insufficient trip instances: at least 3 required'
    from sklearn.model_selection import GroupKFold
    splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(rows, groups=groups))
    return splits, 'Exploratory grouped trip cross-validation; not a future-date performance estimate'


def metric_rows(predictions):
    metrics = []
    sections = {'all': predictions}
    for quality in sorted({r['label_confidence'] for r in predictions}):
        sections['quality:'+quality] = [r for r in predictions if r['label_confidence'] == quality]
    sections['strict_labels_only'] = [r for r in predictions if r['label_confidence'] != 'low_crossing']
    for route in sorted({r['route_id'] for r in predictions}):
        sections['route:'+route] = [r for r in predictions if r['route_id'] == route]
    for section, rows in sections.items():
        if not rows:
            continue
        truth = np.array([r['remaining_travel_time_seconds'] for r in rows])
        lo = np.array([r['remaining_time_lower_seconds'] for r in rows])
        hi = np.array([r['remaining_time_upper_seconds'] for r in rows])
        counts = Counter(r['arrival_event_id'] for r in rows)
        weights = np.array([1/counts[r['arrival_event_id']] for r in rows])
        for method in ('model', 'schedule', 'official_eta'):
            estimates = np.array([r[method+'_prediction_seconds'] for r in rows])
            errors = estimates-truth
            metrics.append(dict(subset=section, method=method, samples=len(rows), arrival_events=len(counts),
                trip_instances=len({r['trip_instance_id'] for r in rows}),
                mae_seconds=float(np.abs(errors).mean()), rmse_seconds=float(np.sqrt(np.mean(errors**2))),
                bias_seconds=float(errors.mean()), event_balanced_mae_seconds=float(np.average(np.abs(errors), weights=weights)),
                interval_distance_mae_seconds=float(np.maximum(np.maximum(lo-estimates, estimates-hi), 0).mean())))
    return metrics


def evaluate_runs(runs, output, quality='all', split='auto'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, exclusions = load_rows(runs, quality)
    predictions, folds, metrics = [], [], []
    status = 'Insufficient data: at least 30 matched samples, 6 arrival events and 3 trip instances are required.'
    protocol = 'No split performed'
    enough = len(rows) >= 30 and len({r['arrival_event_id'] for r in rows}) >= 6 and len({r['trip_instance_id'] for r in rows}) >= 3
    params = dict(n_estimators=150, min_samples_leaf=3, max_depth=12, random_state=42, n_jobs=1)
    if enough:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.ensemble import RandomForestRegressor
        splits, protocol = split_rows(rows, split)
        for fold, (train, test) in enumerate(splits):
            train_groups = {rows[i]['trip_instance_id'] for i in train}
            test_groups = {rows[i]['trip_instance_id'] for i in test}
            if train_groups & test_groups:
                raise ValueError('Train/test trip leakage')
            train_events = {rows[i]['arrival_event_id'] for i in train}
            test_events = {rows[i]['arrival_event_id'] for i in test}
            if train_events & test_events:
                raise ValueError('Train/test arrival-event leakage')
            manifest = dict(fold=fold, train_trip_instances=sorted(train_groups), test_trip_instances=sorted(test_groups),
                            train_samples=len(train), test_samples=len(test), train_events=len(train_events), test_events=len(test_events))
            if len(train) < 10 or len(train_groups) < 2 or len(train_events) < 3 or not len(test):
                manifest['status'] = 'skipped_insufficient_training_data'
                folds.append(manifest)
                continue
            manifest['status'] = 'evaluated'
            folds.append(manifest)
            vectorizer = DictVectorizer(sparse=True)
            xtrain = vectorizer.fit_transform([features(rows[i]) for i in train])
            xtest = vectorizer.transform([features(rows[i]) for i in test])
            ytrain = [rows[i]['remaining_travel_time_seconds'] for i in train]
            event_counts = Counter(rows[i]['arrival_event_id'] for i in train)
            weights = [1/event_counts[rows[i]['arrival_event_id']] for i in train]
            model = RandomForestRegressor(**params)
            model.fit(xtrain, ytrain, sample_weight=weights)
            inferred = model.predict(xtest)
            for i, estimate in zip(test, inferred):
                r = rows[i]
                predictions.append(dict(
                    **{k:r[k] for k in ('sample_id', 'arrival_event_id', 'trip_instance_id', 'service_date',
                        'route_id', 'direction_id', 'vehicle_id', 'target_stop_id', 'observation_timestamp',
                        'label_confidence', 'remaining_travel_time_seconds', 'remaining_time_lower_seconds',
                        'remaining_time_upper_seconds', 'source_run')},
                    fold=fold, model_prediction_seconds=float(estimate),
                    schedule_prediction_seconds=r['scheduled_remaining_seconds'],
                    official_eta_prediction_seconds=r['official_remaining_seconds']))
        metrics = metric_rows(predictions)
        status = f'Exploratory comparison: {len(predictions)} held-out samples, {len({r["arrival_event_id"] for r in predictions})} arrival events.' if predictions else 'Insufficient data for an eligible training/test split.'
    write_csv(output / 'predictions.csv', predictions, ['sample_id', 'fold', 'arrival_event_id', 'trip_instance_id',
              'model_prediction_seconds', 'schedule_prediction_seconds', 'official_eta_prediction_seconds'])
    write_csv(output / 'metrics.csv', metrics, ['subset', 'method', 'samples', 'arrival_events', 'mae_seconds', 'rmse_seconds'])
    write_csv(output / 'excluded_samples.csv', exclusions, ['sample_id', 'source_run', 'reason'])
    config = dict(status=status, protocol=protocol, quality=quality, requested_split=split, parameters=params,
                  feature_names=sorted(features(rows[0])) if rows else [],
                  inputs=[str(Path(r).resolve()) for r in runs], eligible_samples=len(rows),
                  eligible_events=len({r['arrival_event_id'] for r in rows}), eligible_trips=len({r['trip_instance_id'] for r in rows}),
                  excluded_samples=len(exclusions), folds=folds,
                  input_hashes={str(Path(r)/name): hashlib.sha256((Path(r)/name).read_bytes()).hexdigest()
                      for r in runs for name in ('training_samples.csv', 'punctuality_events.csv') if (Path(r)/name).exists()})
    import sklearn
    config['sklearn_version'] = sklearn.__version__
    config['numpy_version'] = np.__version__
    (output / 'evaluation.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    lines = ['# Matched arrival-time prediction comparison', '', status, '', protocol, '',
             f'Eligible matched samples: {len(rows)}; excluded: {len(exclusions)}. Quality filter: {quality}.',
             f'Exclusion reasons: {dict(Counter(r["reason"] for r in exclusions))}', '',
             f'Evaluated folds: {sum(f["status"] == "evaluated" for f in folds)}/{len(folds)}. Skipped folds are listed in evaluation.json and do not contribute scores.', '',
             '| Predictor | Test samples | Events | MAE (s) | RMSE (s) | Event-balanced MAE (s) |',
             '|---|---:|---:|---:|---:|---:|']
    for m in metrics:
        if m['subset'] == 'all':
            lines.append(f'| {m["method"]} | {m["samples"]} | {m["arrival_events"]} | {m["mae_seconds"]:.1f} | {m["rmse_seconds"]:.1f} | {m["event_balanced_mae_seconds"]:.1f} |')
    lines += ['', '## Interpretation and reproducibility', '',
        '- Model: fixed RandomForestRegressor (150 trees, depth 12, leaf size 3, seed 42). No hyperparameter search or test-set tuning. Each training event receives equal total sample weight.',
        '- Model inputs: current GPS location, straight/along-route distance, GPS age, local time/weekday, route, direction, target stop, and static schedule remaining time. Official ETA is deliberately excluded from model inputs.',
        '- Feature encoding is fitted on training folds only. Trip IDs, arrival IDs, label confidence, future GPS and arrival times are never model features.',
        '- Baselines: archived static arrival time and official ETA from the prediction snapshot, each minus the same observation time. Negative baseline remaining times are retained rather than silently clipped.',
        '- A row is eligible only when it has a label, valid model features, a schedule match and official ETA. All three methods use exactly the same held-out rows and labels. Missing-ETA rows are audited separately, not assigned fake predictions.',
        '- Multiple dates use the latest 20% of service dates as test data; training labels that overlap test time are purged. A one-day dataset uses grouped trip cross-validation, which does not measure future-date generalization.',
        '- Entire trip instances and arrival events remain in one side of each split, including across input runs. Repeated input samples are deduplicated. Fold manifests are in evaluation.json.',
        '- Metrics by label confidence, strict-label subset and route are in metrics.csv. Strict-label subset scores use the same fitted models; rerun with --quality strict for a separately trained strict-only sensitivity analysis.',
        '- Targets are uncertain GPS stop-passage estimates, not verified door-opening times. Interval-distance MAE measures distance outside label bounds and is not a substitute for MAE.',
        '- The same GPS/shape information helps construct labels and model features, which can bias apparent accuracy. Validate a sample against independently observed arrivals before claiming superiority.',
        '- These counts are a software feasibility check, not evidence of adequate statistical power. Small numbers of trips, unseen routes and correlated arrivals make rankings unstable. No deployment model is exported; models are fitted independently for evaluation folds.',
        '- No weather features are included. Collect multiple days and reserve a future-date test set before comparing weather ablations.']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    overall = [m for m in metrics if m['subset'] == 'all']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, title in zip(axes, ('mae_seconds', 'rmse_seconds', 'event_balanced_mae_seconds'), ('MAE', 'RMSE', 'Event-balanced MAE')):
        if overall:
            bars = ax.bar(['ML model', 'Schedule', 'Official ETA'], [m[metric]/60 for m in overall], color=['#3579a8', '#939ca6', '#19846b'])
            ax.bar_label(bars, fmt='%.2f min', padding=5)
            ax.set_ylim(0, max(m[metric]/60 for m in overall)*1.22+.1)
        else:
            ax.text(.5, .5, 'Insufficient data\nNo model comparison computed', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
        ax.set(title=title+' (lower is better)', ylabel='Prediction error (minutes)')
    fig.suptitle('Arrival-time predictions: identical held-out samples', fontsize=15)
    fig.text(.5, .015, f'{len(predictions)} test samples | GPS-inferred labels | Exploratory results, not proof of generalization', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93)); fig.savefig(output / 'model_comparison.png', dpi=150); plt.close(fig)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quality', choices=['all', 'strict'], default='all')
    parser.add_argument('--split', choices=['auto', 'date', 'trip'], default='auto')
    args = parser.parse_args()
    if any(not (run/'training_samples.csv').exists() for run in args.runs):
        parser.error('Each input must contain training_samples.csv; run postprocess first')
    print(evaluate_runs(args.runs, args.output, args.quality, args.split))
