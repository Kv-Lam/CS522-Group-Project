"""Validate crossing labels and leakage-safe matched evaluation with synthetic data."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from arrival_rules import crossing_events
from evaluate import features, split_rows, metric_rows, evaluate_runs, write_csv


class CrossingTests(unittest.TestCase):
    def setUp(self):
        self.points = [dict(vehicle_id='v', trip_instance_id='t|20260915|12:00:00', continuity_id='c',
            route_id='r', route_name='Route', direction_id='0', service_date='20260915',
            latitude=36., longitude=-84.+(progress-100)/89958., shape_offset_m=0,
            along_position_m=progress, gps_timestamp=ts, current_stop_sequence=None)
            for progress, ts in [(0, 1000), (200, 1030), (300, 1060)]]
        self.stops = [dict(along_m=100, seq=1, stop_id='s', lat=36., lon=-84.)]

    def test_crossing_without_near_station_point(self):
        event = crossing_events(self.points, self.stops)[0]
        self.assertEqual(event['arrival_timestamp'], 1015)
        self.assertEqual(event['arrival_lower_timestamp'], 985)
        self.assertEqual(event['arrival_upper_timestamp'], 1045)
        self.assertEqual(event['label_confidence'], 'low_crossing')
        self.assertEqual(event['evidence_confirmation_timestamp'], 1060)

    def test_gap_rejected(self):
        self.points[1]['gps_timestamp'] = 1070
        self.points[2]['gps_timestamp'] = 1100
        self.assertFalse(crossing_events(self.points, self.stops))

    def test_change_of_trip_or_segment_rejected(self):
        self.points[2]['continuity_id'] = 'new'
        self.assertFalse(crossing_events(self.points, self.stops))

    def test_no_departure_confirmation_rejected(self):
        self.points[2]['along_position_m'] = 199
        self.assertFalse(crossing_events(self.points, self.stops))

    def test_offroute_and_sequence_conflicts_rejected(self):
        self.points[0]['shape_offset_m'] = 50
        self.assertFalse(crossing_events(self.points, self.stops))
        self.points[0]['shape_offset_m'] = 0
        self.points[0]['current_stop_sequence'] = 2
        self.assertFalse(crossing_events(self.points, self.stops))


class EvaluationTests(unittest.TestCase):
    def row(self):
        return dict(latitude=36, longitude=-84, along_route_distance_m=500,
            straight_line_distance_m=400, gps_age_seconds=2, scheduled_remaining_seconds=100,
            route_id='r', direction_id='0', target_stop_id='s', observation_local='2026-09-15T12:00:00-04:00')

    def test_future_labels_and_eta_are_not_features(self):
        row = self.row()
        first = features(row)
        row.update(remaining_travel_time_seconds=999, official_eta_timestamp=999,
                   arrival_event_id='future', label_confidence='future', trip_instance_id='future')
        self.assertEqual(features(row), first)

    def test_trip_groups_never_overlap(self):
        rows = [dict(trip_instance_id=str(g), service_date='20260915') for g in range(3) for _ in range(3)]
        splits, _ = split_rows(rows)
        seen = []
        for train, test in splits:
            self.assertFalse({rows[i]['trip_instance_id'] for i in train} & {rows[i]['trip_instance_id'] for i in test})
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(9)))

    def test_date_holdout_purges_overlapping_labels(self):
        rows = [dict(trip_instance_id='old', service_date='20260915', observation_timestamp=0, remaining_time_upper_seconds=10),
                dict(trip_instance_id='overlap', service_date='20260915', observation_timestamp=90, remaining_time_upper_seconds=20),
                dict(trip_instance_id='new', service_date='20260916', observation_timestamp=100, remaining_time_upper_seconds=20)]
        splits, _ = split_rows(rows)
        self.assertEqual(list(splits[0][0]), [0])
        self.assertEqual(list(splits[0][1]), [2])

    def test_metrics_weight_events_and_keep_same_rows(self):
        rows = [dict(arrival_event_id=e, trip_instance_id='t', route_id='r', label_confidence='medium',
                remaining_travel_time_seconds=100, remaining_time_lower_seconds=90, remaining_time_upper_seconds=110,
                model_prediction_seconds=p, schedule_prediction_seconds=p, official_eta_prediction_seconds=p)
                for e,p in [('a',100), ('a',100), ('b',160)]]
        overall = [m for m in metric_rows(rows) if m['subset']=='all']
        self.assertTrue(all(m['samples']==3 and m['mae_seconds']==20 and m['event_balanced_mae_seconds']==30 for m in overall))

    def test_end_to_end_shared_test_set_and_missing_eta(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            run = Path(folder)
            samples, events = [], []
            for g in range(3):
                for event in range(3):
                    event_id = f'{g}|{event}'
                    events.append(dict(arrival_event_id=event_id, schedule_match_status='matched', scheduled_arrival_timestamp=1200))
                    for j in range(5):
                        samples.append(dict(self.row(), sample_id=f'{event_id}|{j}', arrival_event_id=event_id,
                            trip_instance_id=f'{g}|20260915|12:00:00', service_date='20260915', vehicle_id=str(g),
                            observation_timestamp=1000, official_eta_timestamp=1210 if j else '',
                            label_confidence='medium', remaining_travel_time_seconds=220+g,
                            remaining_time_lower_seconds=200, remaining_time_upper_seconds=240))
            write_csv(run/'training_samples.csv', samples, [])
            write_csv(run/'punctuality_events.csv', events, [])
            status = evaluate_runs([run], run/'evaluation')
            self.assertIn('36 held-out samples', status)
            manifest = json.loads((run/'evaluation'/'evaluation.json').read_text())
            self.assertEqual(manifest['excluded_samples'], 9)
            self.assertEqual(len(manifest['folds']), 3)
            with (run/'evaluation'/'predictions.csv').open(encoding='utf-8-sig') as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len({r['sample_id'] for r in predictions}), 36)
            self.assertTrue(all(r['model_prediction_seconds'] and r['official_eta_prediction_seconds'] and r['schedule_prediction_seconds'] for r in predictions))


if __name__ == '__main__':
    unittest.main()
