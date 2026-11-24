"""Sweep the XGBoost late penalty and show the tradeoff on fixed yardsticks.

Run from the folder that contains main.py and evaluate.py:
    python sweep_penalty.py runs/<run_folder> --output runs/<run_folder>/penalty_sweep

Why fixed yardsticks: a model trained at penalty p always scores best at penalty p, so
"lowest late-penalized RMSE" just picks whatever you scored with. Plain MAE, bias and the
share of late predictions do not depend on p, so they show the real tradeoff. Choose the
penalty from that table (for example, the smallest p that reaches a late-rate target, or
where MAE starts to climb steeply), and decide before looking at final test results.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_runs  # noqa: E402

YARDSTICKS = (1.0, 2.0, 4.0)  # penalties used only for scoring, never for training


def read_predictions(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(rows, key):
    truth = np.array([float(r['remaining_travel_time_seconds']) for r in rows])
    errors = np.array([float(r[key]) for r in rows]) - truth  # positive = predicted later than the label
    result = dict(mae_s=float(np.abs(errors).mean()), bias_s=float(errors.mean()),
                  late_rate=float((errors > 0).mean()), mean_late_s=float(np.maximum(errors, 0).mean()))
    for p in YARDSTICKS:
        result[f'pen_rmse_{p:g}'] = float(math.sqrt(np.mean(np.where(errors > 0, p, 1.0) * errors ** 2)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--penalties', type=float, nargs='+', default=[1, 1.5, 2, 3, 4, 6, 8, 10, 12])
    parser.add_argument('--quality', choices=['all', 'strict'], default='all')
    parser.add_argument('--split', choices=['auto', 'date', 'trip'], default='auto')
    args = parser.parse_args()

    table, baselines = [], {}
    for penalty in args.penalties:
        out = args.output / f'penalty_{penalty:g}'
        status = evaluate_runs(args.runs, out, args.quality, args.split, model='xgboost', late_penalty=penalty)
        rows = read_predictions(out / 'predictions.csv')
        if not rows:
            raise SystemExit(f'No held-out predictions: {status}')
        if not baselines:  # same held-out rows for every penalty, so score the baselines once
            baselines = {'schedule': summarize(rows, 'schedule_prediction_seconds'),
                         'official ETA': summarize(rows, 'official_eta_prediction_seconds')}
            n_samples, n_events = len(rows), len({r['arrival_event_id'] for r in rows})
        table.append(dict(train_penalty=penalty, **summarize(rows, 'xgboost_prediction_seconds')))

    header = (f'{"model":>16} | {"MAE":>6} {"bias":>7} {"late%":>6} {"meanLate":>8} | '
              + ' '.join(f'{"pRMSE@" + format(p, "g"):>9}' for p in YARDSTICKS))

    def line(label, r):
        return (f'{label:>16} | {r["mae_s"]:6.1f} {r["bias_s"]:+7.1f} {r["late_rate"]:6.1%} {r["mean_late_s"]:8.1f} | '
                + ' '.join(f'{r[f"pen_rmse_{p:g}"]:9.1f}' for p in YARDSTICKS))

    print(f'\nHeld-out rows: {n_samples} samples, {n_events} arrival events (times in seconds)')
    print(header)
    for row in table:
        print(line(f'XGBoost p={row["train_penalty"]:g}', row))
    for name, row in baselines.items():
        print(line(name, row))
    print('\nCompare rows within a column. MAE, bias, late% and meanLate do not depend on the penalty used for scoring.')

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'sweep_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['model'] + list(table[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow(dict(row, model='xgboost'))
        for name, row in baselines.items():
            writer.writerow(dict(row, model=name))
    print(f'Wrote {args.output / "sweep_summary.csv"}')


if __name__ == '__main__':
    main()