# Matched arrival-time prediction comparison

Exploratory comparison: 43205 held-out samples, 5451 arrival events.

Chronological service-date holdout; overlapping label intervals purged

Eligible matched samples: 155063; excluded: 11045. Quality filter: all.
Exclusion reasons: {'Missing official ETA; excluded from common comparison': 11045}

Evaluated folds: 1/1. Skipped folds are listed in evaluation.json and do not contribute scores.

| Predictor | Test samples | Events | MAE (s) | RMSE (s) | Event-balanced MAE (s) | Late predictions | Late-penalized RMSE (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random forest | 43205 | 5451 | 75.7 | 182.5 | 63.1 | 48.7% | 535.4 |
| XGBoost | 43205 | 5451 | 81.1 | 140.8 | 46.5 | 21.1% | 193.4 |
| Schedule | 43205 | 5451 | 554.9 | 2625.9 | 729.2 | 28.0% | 8559.5 |
| Official ETA | 43205 | 5451 | 104.8 | 456.9 | 82.3 | 35.7% | 1529.7 |

## Interpretation and reproducibility

- Late means the predicted arrival is after the GPS label. Late-penalized RMSE multiplies squared positive errors by 12 for every predictor before averaging and taking the square root. Ordinary MAE and RMSE still penalize either direction equally.
- Model settings are fixed before evaluation. Each training arrival event receives equal total sample weight before any direction-specific penalty.
- Model inputs: current GPS location, straight/along-route distance, GPS age, local time/weekday, route, direction, target stop, and static schedule remaining time. Official ETA is deliberately excluded from model inputs.
- Feature encoding is fitted on training folds only. Trip IDs, arrival IDs, label confidence, future GPS and arrival times are never model features.
- Baselines: archived static arrival time and official ETA from the prediction snapshot, each minus the same observation time. Negative baseline remaining times are retained rather than silently clipped.
- A row is eligible only when it has a label, valid model features, a schedule match and official ETA. All predictors use exactly the same held-out rows and labels. Missing ETA rows are audited separately.
- Multiple dates use the latest 20% of service dates as test data; training labels that overlap test time are purged. A one-day dataset uses grouped trip cross-validation, which does not measure future-date generalization.
- Entire trip instances and arrival events remain in one side of each split, including across input runs. Repeated input samples are deduplicated. Fold manifests are in evaluation.json.
- Metrics by label confidence, strict-label subset and route are in metrics.csv. Strict-label subset scores use the same fitted models; rerun with --quality strict for a separately trained strict-only sensitivity analysis.
- Targets are uncertain GPS stop-passage estimates, not verified door-opening times. Interval-distance MAE measures distance outside label bounds and is not a substitute for MAE.
- The same GPS/shape information helps construct labels and model features, which can bias apparent accuracy. Validate a sample against independently observed arrivals before claiming superiority.
- These counts are a software feasibility check, not evidence of adequate statistical power. Small numbers of trips, unseen routes and correlated arrivals make rankings unstable. No deployment model is exported; models are fitted independently for evaluation folds.
- No weather features are included. Collect multiple days and reserve a future-date test set before comparing weather ablations.
- XGBoost trains with asymmetric squared error, weighting positive errors 12×. Its initial score uses the training fold only. Predicted remaining times are floored at zero.
- Random forest uses the original settings and symmetric squared-error training. It remains the `model` method in the CSV files.
