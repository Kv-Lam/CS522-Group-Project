"""Synthetic rule tests; fixtures are never mixed with collected research data."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from main import Processor, instance, pb, outputs, read_jsonl, collect


class ArrivalRulesTests(unittest.TestCase):
    def setUp(self):
        self.p = Processor(None)
        self.stop = {'seq': 4, 'stop_id': 'test_stop', 'along_m': 100, 'lat': 36., 'lon': -84.}
        self.points = [dict(vehicle_id='test_vehicle', trip_instance_id='trip|20260915|25:00:00',
                            continuity_id='segment', route_id='test_route', route_name='Test Route',
                            direction_id='0', service_date='20260915',
                            gps_timestamp=t, along_position_m=progress, latitude=36., longitude=lon,
                            current_stop_sequence=None, vehicle_status=None)
                       for t, progress, lon in [(1000, 0, -84.001), (1030, 100, -84.), (1060, 200, -83.999)]]

    def test_three_point_arrival_and_deduplication(self):
        self.p.detect_arrivals(self.points, [self.stop])
        self.p.detect_arrivals(self.points, [self.stop])
        self.assertEqual(len(self.p.events), 1)
        event = next(iter(self.p.events.values()))
        self.assertEqual(event['arrival_timestamp'], 1030)
        self.assertEqual(event['label_source'], 'gps_inferred')
        self.assertEqual(event['time_uncertainty_seconds'], 45)

    def test_single_near_point_without_departure_is_not_arrival(self):
        self.points[2]['along_position_m'] = 105
        self.p.detect_arrivals(self.points, [self.stop])
        self.assertFalse(self.p.events)

    def test_gap_is_not_arrival(self):
        self.points[2]['gps_timestamp'] = 1200
        self.p.detect_arrivals(self.points, [self.stop])
        self.assertFalse(self.p.events)

    def test_duplicate_timestamp_is_not_arrival(self):
        self.points[1]['gps_timestamp'] = 1000
        self.p.detect_arrivals(self.points, [self.stop])
        self.assertFalse(self.p.events)

    def test_wrong_sequence_is_not_arrival(self):
        self.points[0]['current_stop_sequence'] = 5
        self.p.detect_arrivals(self.points, [self.stop])
        self.assertFalse(self.p.events)

    def test_midnight_instance_and_date_separation(self):
        descriptor = pb.TripDescriptor(trip_id='trip', start_date='20260915', start_time='25:00:00')
        first = instance(descriptor)
        self.assertIn('25:00:00', first)
        descriptor.start_date = '20260916'
        self.assertNotEqual(first, instance(descriptor))

    def test_missing_instance_date_is_rejected(self):
        with self.assertRaises(ValueError):
            instance(pb.TripDescriptor(trip_id='trip', start_time='10:00:00'))

    def test_truncated_journal_is_recoverable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'journal.jsonl'
            path.write_text('{"ok": true}\n{"partial":', encoding='utf-8')
            self.assertEqual(read_jsonl(path), [{'ok': True}])

    def test_interrupt_finalizes_and_calls_postprocess(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            run = Path(folder) / 'interrupted_run'
            args = SimpleNamespace(output=str(run), duration_minutes=60, interval=30, timeout=1, retries=0)
            with patch('main.fetch', side_effect=KeyboardInterrupt), patch('main.postprocess', return_value=None) as replay:
                self.assertEqual(collect(args), 0)
                replay.assert_called_once_with(run, early=60, late=300)
            self.assertIn('interrupted', (run / 'run.json').read_text())

    def test_future_predictions_are_not_labels_and_gaps_excluded(self):
        self.p.detect_arrivals(self.points, [self.stop])
        base = dict(sample_id='sample', vehicle_id='test_vehicle',
                    trip_instance_id=self.points[0]['trip_instance_id'], continuity_id='segment',
                    target_stop_sequence=4, observation_timestamp=900,
                    official_eta_timestamp=1100, official_eta_status='available')
        self.p.samples = [base, dict(base, sample_id='gap', continuity_id='other'),
                          dict(base, sample_id='overlap', observation_timestamp=1000),
                          dict(base, sample_id='no_event', target_stop_sequence=5)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            run = Path(folder)
            outputs(run, self.p, [], [])
            import csv
            with (run / 'training_samples.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]['remaining_travel_time_seconds']), 130)
            self.assertEqual(float(rows[0]['official_eta_timestamp']), 1100)
            with (run / 'excluded_records.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)


class DynamicSelectionTests(unittest.TestCase):
    def observation(self, vehicle_id, timestamp, progress, route='route'):
        return dict(vehicle_id=vehicle_id, trip_id='trip', gps_timestamp=timestamp,
                    along_position_m=progress, trip_instance_id=f'{vehicle_id}|trip',
                    route_id=route, route_name=route, direction_id='0')

    def test_new_vehicle_is_discovered_after_initial_selection(self):
        processor = Processor(None)
        processor.discover([self.observation('a', 100, 0)])
        processor.discover([self.observation('a', 130, 50)])
        self.assertEqual([row['vehicle_id'] for row in processor.selected], ['a'])

        processor.discover([self.observation('a', 160, 100), self.observation('b', 160, 0)])
        processor.discover([self.observation('a', 190, 150), self.observation('b', 190, 50)])
        self.assertEqual([row['vehicle_id'] for row in processor.selected], ['a', 'b'])

    def test_inactive_selection_is_retained_and_active_slot_refilled(self):
        processor = Processor(None)
        processor.discover([self.observation('a', 100, 0), self.observation('b', 100, 0)])
        processor.discover([self.observation('a', 130, 50), self.observation('b', 130, 50)])
        self.assertEqual({row['vehicle_id'] for row in processor.selected}, {'a', 'b'})

        processor.discover([self.observation('c', 160, 0), self.observation('b', 160, 100)])
        processor.discover([self.observation('c', 190, 50), self.observation('b', 190, 150)])
        self.assertEqual({row['vehicle_id'] for row in processor.selected}, {'a', 'b', 'c'})

    def test_disappearing_vehicle_selection_and_history_are_preserved(self):
        processor = Processor(None)
        processor.selected.append(dict(route_id='route', route_name='route', direction_id='0', vehicle_id='a'))
        processor.history['segment'].append({'vehicle_id': 'a', 'gps_timestamp': 100})

        processor.discover([])

        self.assertEqual([row['vehicle_id'] for row in processor.selected], ['a'])
        self.assertEqual(len(processor.history['segment']), 1)

    def test_repeated_discovery_does_not_duplicate_selection(self):
        processor = Processor(None)
        processor.discover([self.observation('a', 100, 0)])
        processor.discover([self.observation('a', 130, 50)])
        processor.discover([self.observation('a', 160, 100)])
        self.assertEqual([row['vehicle_id'] for row in processor.selected], ['a'])

    def test_selected_vehicle_is_not_followed_to_another_route(self):
        processor = Processor(None)
        processor.selected.append(dict(route_id='route', route_name='route', direction_id='0', vehicle_id='a'))
        processor.discover([self.observation('a', 100, 0, route='other')])
        processor.discover([self.observation('a', 130, 50, route='other')])
        self.assertEqual(processor.selected, [dict(route_id='route', route_name='route', direction_id='0', vehicle_id='a')])

    def test_discovery_preserves_two_active_route_limit(self):
        processor = Processor(None)
        processor.selected.extend([
            dict(route_id='route1', route_name='route1', direction_id='0', vehicle_id='a'),
            dict(route_id='route2', route_name='route2', direction_id='0', vehicle_id='b')])
        processor.discover([self.observation('a', 100, 0, route='route1'),
                            self.observation('b', 100, 0, route='route2'),
                            self.observation('c', 100, 0, route='route3')])
        processor.discover([self.observation('a', 130, 50, route='route1'),
                            self.observation('b', 130, 50, route='route2'),
                            self.observation('c', 130, 50, route='route3')])
        self.assertEqual({row['route_id'] for row in processor.selected}, {'route1', 'route2'})

    def test_duplicate_observation_timestamp_is_rejected(self):
        processor = Processor(SimpleNamespace(pattern=lambda _: []))
        first = self.observation('a', 100, 0)
        first.update(observation_timestamp=100, latitude=36., longitude=-84.)
        duplicate = dict(first)
        processor.process_observation(first, None)
        processor.process_observation(duplicate, None)
        self.assertEqual(len(processor.history[first['continuity_id']]), 1)
        self.assertEqual(processor.excluded[-1]['reason'], 'Duplicate or out-of-order vehicle timestamp')


if __name__ == '__main__':
    unittest.main()
