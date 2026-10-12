# KAT Bus Arrival Prediction: One-Hour Data Experiment

This project collects real KAT GTFS and GTFS-Realtime Protobuf data, automatically selects routes, creates prediction samples, infers stop arrival/pass-by times, and compares machine learning, the static schedule, and official ETA on the same test samples. It does not use weather or event data and does not treat official ETA as ground truth.

## Environment and startup

Python 3.10 or newer is required; Python 3.12 is recommended. Collection requires network access to KAT GTFS/GTFS-Realtime services; processing an existing run directory does not require network access.

Run in PowerShell:

```powershell
Set-Location 'C:\path\to\kat_arrival_experiment'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the `py` launcher is unavailable, use an installed Python 3.10+ executable to create the virtual environment. If PowerShell blocks the activation script, run `.\.venv\Scripts\python.exe` directly, for example: `& .\.venv\Scripts\python.exe .\main.py collect`.

Start collection after installation:

```powershell
python .\main.py collect
```

The default run lasts 60 minutes with one round every 30 seconds. Static GTFS initialization and final postprocessing are additional time. Requests are serial; each request has a 15-second timeout and at most one retry by default. Slow networks can increase the actual interval, and the report records gaps. Keep the machine connected and awake during collection; no system settings need to be changed.

```powershell
# Short collection test
python .\main.py collect --duration-minutes 2
# Custom duration and sampling interval
python .\main.py collect --duration-minutes 15 --interval 30
# Replay an existing collection without network access
python .\main.py postprocess .\runs\short_test
# Run isolated synthetic rule tests
python -m unittest -v test_rules
```

## Sharing with Teammates

When sharing runnable source, include at least `main.py`, `arrival_rules.py`, `punctuality.py`, `evaluate.py`, `requirements.txt`, and this README. Including `.gitignore`, `test_rules.py`, `test_punctuality.py`, `test_improvements.py`, and `TEST_RESULTS.md` is recommended; `probe.py` is only an optional live-data inspection tool.

After installing the dependencies in `requirements.txt`, teammates can run the project without packaging `vendor/`. New runs download the required GTFS data automatically, so `probe_data/` and `runs/` usually do not need to be included. These directories may contain large raw feeds and vehicle-position records; confirm that the data is suitable for distribution before sharing. Offline reprocessing requires the complete `runs/<run-directory>/`, not only its report or CSV files. Temporary directories, `.mplconfig/`, `__pycache__/`, and `tmp.../` are not deliverables.

The `runs/short_test` and timestamped run directories mentioned below are historical examples and are not automatically included in a minimal source package. Replace them with a run directory that you already have or have just collected.

Pressing Ctrl+C once ends collection and starts postprocessing. Wait for saving to finish; if the process is forcibly terminated or postprocessing is interrupted again, use the standalone postprocessing command to recover. Raw data and request logs are written each round, and an incomplete final log line is ignored.

## Automatic Route Selection and Matching

Each run downloads static GTFS again and matches `trip_id`, direction, route, stop order, and shape. It first observes two fresh vehicle positions to confirm forward movement, then selects up to two routes by the number of trackable vehicles; each route has one direction and at most two vehicles, for no more than four vehicles total. If only one route is found, it uses that route; if no reliable candidate exists, collection continues and the report says so.

Each interval rechecks the current vehicle feed for newly moving, trackable buses using the same two-observation movement rule. Up to two routes and two active vehicles per route are considered in each discovery pass; previously selected vehicles remain monitored when present, and later eligible buses can fill a slot when a selected vehicle disappears. Selections and collected records are retained for the run, while a missing bus simply contributes no new observations. A vehicle changing trips starts a new trip-instance segment; a change to another route or direction is not analyzed. Non-loop filtering is conservative: trips with repeated stops, an end-to-end distance under 500 meters, a route name containing loop/circular, or non-monotonic or ambiguous stop projections are excluded. This may exclude usable short or complex routes.

The trip instance is `trip_id|start_date|start_time`; the GTFS service date and departure times beyond 24:00 are preserved, and missing instance fields are never guessed. All label arithmetic uses UTC seconds, while local time uses the static GTFS `America/New_York` timezone. Events are not merged across dates, midnight, or trip changes.

## Samples and Labels

Each valid vehicle observation creates samples only for the three nearest stops more than 20 meters downstream on the current shape. The observation time is the receipt time obtained for that round; GPS time and data age are stored separately. Official ETA comes only from an explicit `arrival.time` in that round's TripUpdates with matching vehicle, trip instance, and stop sequence; it is left blank when missing, stale, canceled, or unmatched. Later ETA updates are never backfilled into features.

Straight-line distance uses haversine; along-route distance is computed from GTFS shape geometry using a local meter approximation, without assuming the unit of GTFS `shape_dist_traveled`. Records with ambiguous projections or more than 80 meters from the shape are not used.

This source has no validated actual-arrival field. Even when a TripUpdates prediction is in the past or `uncertainty=0`, it is not treated as a direct observation. The implementation uses **GPS-inferred stop pass-by events**:

1. Three observations with increasing timestamps from the same vehicle, trip instance, and continuous segment; adjacent observations are no more than 75 seconds apart.
2. The first observation is at least 35 meters before the stop position; the second is within 45 meters of the stop and has at most 45 meters of along-route error; the third is at least 35 meters past the stop position.
3. The first and third observations are each at least 15 meters farther from the stop than the second; available stop order must not contradict movement.
4. If the middle point is explicitly `STOPPED_AT` for the matching stop sequence, rule confidence is `medium_status_supported`; otherwise it is `medium`. This is not a calibrated probability and does not guarantee a stop or door opening.

The event time is the middle GPS time. The conservative interval covers the first through third points and is expanded by 15 seconds at both ends to allow for source and local clock differences. In this experiment the source was about 5–6 seconds ahead of the local clock; the program allows up to 15 seconds of lead, preserves negative GPS ages explicitly, and does not change system time. A sample must be strictly before the entire event interval and in the same continuous segment to receive `remaining_travel_time_seconds = arrival_timestamp - observation_timestamp`. Label bounds and uncertainty are also stored.

Duplicate snapshots, non-increasing vehicle times, data older than 120 seconds, gaps over 75 seconds, trip changes, significant reversals, and implausible jumps cannot be crossed when generating labels. Only one event is recorded for each vehicle-trip stop sequence. Samples that cannot be determined, do not pass a stop, end in a truncated segment, or are marked as skipped by the source go to the exclusion table; labels are never forced. Thirty-second sampling can easily miss nearby stops, and conservative rules may produce few events.

## Output and Reprocessing

Default output is `runs\\<UTC-timestamp>\\`; each run creates a new directory, and `--output` can specify a directory that does not yet exist.

| File | Contents |
|---|---|
| `raw/` | Static ZIP and full-network raw Protobuf data for reprocessing |
| `requests.jsonl` | Request status, receipt time, checksum, feed time, errors, and limited retries for each request |
| `ticks.jsonl` | Requests used in each round and raw file paths |
| `run.json` | Run parameters, start/end times, and final status |
| `selection.json` | Automatically selected routes, directions, and vehicle IDs |
| `observations.csv` | Quality-filtered observations from selected vehicles |
| `arrival_events.csv` | Independent arrival events, source, rule, confidence, and time interval |
| `training_samples.csv` | Labeled samples, prediction-time features, and quality flags only |
| `excluded_records.csv` | Unlabeled samples or excluded selected-vehicle records and reasons |
| `report.md` | Interface success, updates, coverage, missing data, event/sample counts, label quality, and ETA error |
| `vehicle_progress.png` | Scatter plot of selected-vehicle position coverage |
| `dataset_quality.png` | Event, labeled-sample, and unlabeled-sample counts |
| `collection.log` | English collection progress and route-selection log |

The report evaluates ETA separately for `direct_observed` and `gps_inferred`; errors are calculated only when each class has at least 5 matched samples covering at least 3 independent arrival events, otherwise data insufficiency is stated explicitly. A positive error means the official predicted time is later than the label; GPS-inferred results themselves have time uncertainty. Multiple samples may share one event, so sample counts are not independent arrival counts.

The formal analysis includes only selected vehicles; route selection may inspect the full-network feed, which is retained as raw data. Reprocessing rebuilds selection, time-of-observation features, and labels in original time order, overwriting derived CSV files, reports, and charts without downloading again. Existing data can be reused after changing rules, but different analysis versions should be saved in separate directories.

Model evaluation is split by service date or trip instance; one arrival event cannot cross the training and test sets. The new version runs the common-sample comparison described below by default.

Reference specification: https://gtfs.org/documentation/realtime/reference/

## Scheduled Timetable Punctuality Analysis

After collection ends or `postprocess` runs, the following are also generated automatically:

| File | Contents |
|---|---|
| `punctuality_events.csv` | Scheduled arrival time, delay seconds, time bounds, early/on-time/late/uncertain classification, and reasons for unmatched events |
| `route_punctuality.csv` | Event counts, trip counts, on-time rates, and uncertainty bounds grouped by route, direction, label source, and schedule precision |
| `delay_distribution.png` | English event-by-event arrival chart with one panel per route and event counts; delay minutes label each point, short bars show time uncertainty, and the green background shows the on-time window |
| `route_punctuality.png` | English route overview showing counts and shares of on-time, late, early, and uncertain events, such as "5 / 8 on time" |
| `route_punctuality_overview.csv` | Route/direction summary matching the overview chart; the on-time rate counts only events that are on time across the entire interval |

Default research definition: **up to 60 seconds early and up to 300 seconds late, including both boundaries, is on time**. This is not a confirmed KAT official performance standard and can be customized:

```powershell
# Collect using a custom on-time window: no early arrival, up to 3 minutes late
python .\main.py collect --early-seconds 0 --late-seconds 180
# Reprocess the completed one-hour run with the default research window
python .\main.py postprocess .\runs\20260915T193508_851091Z
# Run all rule and schedule comparison tests
python -m unittest -v test_rules test_punctuality
```

Reprocessing restores the 60/300-second thresholds by default; pass custom values again to retain them. The thresholds actually used are stored in `processing_config.json`.

Delay is **GPS-inferred/directly observed arrival time minus the static GTFS scheduled `arrival_time`**. Official ETA is not used as truth, and `departure_time` does not replace the target stop's `arrival_time`. Events are matched by trip, service date, stop sequence, and stop; events that cannot be mapped reliably because of a mismatched trip-instance time, missing schedule time, frequency-based service, or similar issues are excluded. Service-day times use the GTFS conversion of local noon minus 12 actual hours, supporting times beyond 24:00 and daylight-saving transitions.

The statistical unit is the deduplicated arrival event, not a training sample sharing its label; multiple stops on one trip may still be related. `point_on_time_rate` uses the estimated event time; its bounds are the shares of events that are on time across the entire interval and possibly on time within the interval. Events crossing a category boundary are `uncertain` and remain in the denominator; these ranges are not statistical confidence intervals. With no matched events, the result shows N/A rather than 0% on time.

`exact_timepoint` is an exact timetable timepoint, while `estimated_stop_time` is an estimated/interpolated stop time; both are reported separately. The uncertainty of estimated schedule times is unknown and is not included in the error range. This analysis covers only successfully identified stop pass-by events from selected vehicles; it does not represent all trips, missed arrivals, canceled trips, or route-wide performance. GPS cannot confirm a door-opening stop.

Charts use English and direct count labels for team sharing. Arrival details are faceted by route/direction with a shared x-axis and an event count for every panel; a route with no reliable arrival events still gets a no-data panel. Circles represent estimated schedule times and squares represent exact timepoints. The route overview combines both time types and states their counts below the chart. Grouped research data remains in `route_punctuality.csv` and the report. The overview rate uses the conservative count of definitely on-time events, while `point_on_time_rate` in the grouped table uses the point-time estimate; boundary-uncertain events can make the two differ.

In this one-hour dataset, Route 11 vehicle 7492 changed to a direction-1 trip at about 15:53 local time; because this run selected only direction 0, later reverse-direction data is not analyzed. Its valid observations cover about 15:35–15:53, with 36 observations and 1 arrival event; Route 16 has 134 valid observations and 8 arrival events. Charts show successfully identified events, not all GPS positions, and cannot be used to compare overall service performance between the routes.

Charts were generated from the `20260915T193508_851091Z` one-hour dataset: all 9 arrival events matched successfully (8 estimated stop times and 1 exact timepoint). `short_test` has no arrival events, and new charts clearly show insufficient data.

Schedule-time interpretation follows the [GTFS Schedule Reference](https://gtfs.org/documentation/schedule/reference/).

## New Version: Enhanced Labels and Fair Model Comparison

The new version uses the `enhanced` label mode by default: it retains the three-point near-stop rules (`medium` / `medium_status_supported`) and adds the low-confidence `low_crossing` rule. The latter does not require a near-stop point but must satisfy:

- Three increasing GPS points from the same vehicle, trip, and continuous segment; the first two cross the stop and the third confirms continued forward movement.
- Adjacent gaps of at most 60 seconds; 50–600 meters of along-route movement for the first two points, with average speed at most 22 m/s; 25–350 meters before and after the stop.
- All three points within 35 meters of the route shape; the stop within 60 meters of the line between the first two points; straight-line/along-route displacement ratio from 0.55 to 1.5; available stop order must not conflict.
- Arrival time is interpolated by along-route progress; the label interval is the GPS time from before to after the stop, expanded by 15 seconds at both ends. Interpolated seconds are not presented as exact ground truth.
- The original strict rules take priority; each event gets at most one label. New evidence times, before/after distances, and rule version are retained in `arrival_events.csv`. `training_samples.csv` contains `label_rule` and confidence.

All labels remain **GPS-inferred stop arrival/pass-by times**. This matches the experiment's goal of approximating actual arrivals with GPS, but requires independent auditing and cannot be claimed to confirm a stop or door opening. Gaps, trip changes, and ambiguous tracks still do not receive labels. More labels do not mean real-world accuracy has been validated.

After normal collection completion or Ctrl+C, postprocessing now runs evaluation automatically; inputs can also be reprocessed offline:

```powershell
# Enhanced labels and automatic matched model evaluation
python .\main.py postprocess .\runs\20260915T193508_851091Z
# Train/evaluate again, keeping only strict labels; the original data is unchanged
python .\evaluate.py .\runs\20260915T193508_851091Z --quality strict --output .\runs\20260915T193508_851091Z\evaluation_strict
# Combine multiple completed, postprocessed runs
python .\evaluate.py .\runs\20260915T193508_851091Z .\runs\short_test --output .\runs\combined_evaluation
# Recreate the old strict labeling only (overwrites derived outputs in this run)
python .\main.py postprocess .\runs\20260915T193508_851091Z --label-mode strict
```

`evaluate.py` uses scikit-learn; version 1.9.1 is installed in the current environment and has been added to `requirements.txt`. Standalone evaluation does not reconnect to the network. Duplicate `sample_id` values across runs are deduplicated and recorded, and each complete `trip_instance_id` stays on one side of the split.

### Evaluation Protocol

1. The model predicts `remaining_travel_time_seconds`. A fixed random forest is used (150 trees, depth 12, minimum leaf size 3, random seed 42), with no test-set tuning. Each training arrival event has equal total sample weight.
2. Only observation-time features are used: latitude/longitude, straight-line/along-route distance, GPS age, local time, weekday, route, direction, target stop, and scheduled remaining time. **Official ETA, labels, future positions, event IDs, trip IDs, and confidence are not model inputs.** Feature encoding is fitted only on training folds.
3. Comparison data must have a valid label, features, static schedule time, and official ETA from that time. Any missing item is excluded uniformly; methods cannot use different test sets. Schedule time and official ETA are not clipped at zero.
4. With two or more service dates, the final 20% of dates are used for testing and earlier dates for training; training samples whose label intervals overlap the test start are removed. With one date, trip-grouped cross-validation is run only with at least three trip instances; this is exploratory and does not represent future-date performance.
5. At least 30 common samples, 6 events, and 3 trips are required; each valid training fold needs at least 10 samples, 3 events, and 2 trips. These are run thresholds, not claims of statistical sufficiency. If unmet, the output states insufficient data and does not fall back to a leakage-prone random sample split. Skipped folds are listed.
6. Each test sample retains model, schedule, and official ETA predictions together. The report includes MAE, RMSE, bias, event-balanced MAE, and distance to the label interval. Results are also reported by route, confidence, and the original strict-label subset; these subsets use the same fitted models. Strict labels can be trained separately with `--quality strict` above.

### New Outputs

Each run directory contains `evaluation/`:

| File | Contents |
|---|---|
| `report.md` | Common-sample comparison of the three methods, split method, and limitations |
| `model_comparison.png` | English comparison of MAE, RMSE, and event-balanced MAE |
| `predictions.csv` | Three predictions, label interval, fold, and event/trip IDs for each test sample |
| `metrics.csv` | Overall evaluation metrics and metrics by route and label confidence |
| `evaluation.json` | Parameters, input checksums, software versions, feature list, training/test trip lists, and skipped folds |
| `excluded_samples.csv` | Samples excluded from common comparison and reasons |

Evaluation trains one model per fold and does not export a deployment model. The current data has only 3 trips, multiple related events on the same route, and shared information between GPS features and inferred labels, creating a risk of bias. More days of data, future-date holdouts, and independent arrival-label audits are needed before claiming an advantage over official ETA; the weather experiment remains future work.
