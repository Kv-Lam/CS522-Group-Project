"""Synthetic schedule tests are isolated from research observations."""
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from punctuality import scheduled_epoch, compare_events, summarize


class PunctualityTests(unittest.TestCase):
    def setUp(self):
        self.zone = ZoneInfo('America/New_York')
        self.static = SimpleNamespace(zone=self.zone, frequency_trips=set(),
            trips={'t': {'route_id': 'r', 'direction_id': '0'}},
            times={'t': [dict(seq=0, stop_id='origin', departure_time='23:30:00'),
                         dict(seq=1, stop_id='s', arrival_time='25:00:00', timepoint='0')]})
        self.scheduled = scheduled_epoch('20260915', '25:00:00', self.zone)

    def event(self, delay=0, uncertainty=0):
        return dict(arrival_event_id='e', trip_instance_id='t|20260915|23:30:00',
            route_id='r', route_name='Route', direction_id='0', service_date='20260915',
            target_stop_sequence=1, target_stop_id='s', label_source='gps_inferred',
            arrival_timestamp=self.scheduled+delay,
            arrival_lower_timestamp=self.scheduled+delay-uncertainty,
            arrival_upper_timestamp=self.scheduled+delay+uncertainty)

    def test_cross_midnight(self):
        self.assertEqual(datetime.fromtimestamp(self.scheduled, self.zone).isoformat(), '2026-09-16T01:00:00-04:00')

    def test_dst_noon_anchor(self):
        for day in ('20260308', '20261101'):
            noon = datetime.strptime(day, '%Y%m%d').replace(hour=12, tzinfo=self.zone)
            self.assertEqual(scheduled_epoch(day, '12:00:00', self.zone), noon.timestamp())
            self.assertEqual(scheduled_epoch(day, '25:00:00', self.zone)-scheduled_epoch(day, '01:00:00', self.zone), 86400)

    def test_inclusive_thresholds_and_delay_sign(self):
        for delay, expected in [(-61, 'early'), (-60, 'on_time'), (300, 'on_time'), (301, 'late')]:
            row = compare_events([self.event(delay)], self.static)[0]
            self.assertEqual(row['punctuality_status'], expected)
            self.assertEqual(row['delay_seconds'], delay)
            self.assertEqual(row['schedule_precision'], 'estimated_stop_time')

    def test_uncertainty_kept_in_denominator_and_duplicates_removed(self):
        event = self.event(300, 45)
        rows = compare_events([event, event], self.static)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['punctuality_status'], 'uncertain')
        summary = summarize(rows, [])[0]
        self.assertEqual(summary['matched_events'], 1)
        self.assertEqual(summary['point_on_time_rate'], 1)
        self.assertEqual(summary['on_time_rate_lower'], 0)
        self.assertEqual(summary['on_time_rate_upper'], 1)

    def test_missing_and_frequency_schedules_excluded(self):
        self.static.times['t'][1]['arrival_time'] = ''
        self.assertEqual(compare_events([self.event()], self.static)[0]['schedule_match_status'], 'excluded')
        self.static.frequency_trips.add('t')
        self.assertIn('Frequency', compare_events([self.event()], self.static)[0]['schedule_exclusion_reason'])

    def test_mismatched_instance_not_shifted(self):
        event = dict(self.event(), trip_instance_id='t|20260915|23:35:00')
        self.assertEqual(compare_events([event], self.static)[0]['schedule_match_status'], 'excluded')

    def test_no_data_rate_is_missing_not_zero(self):
        summary = summarize([], [dict(route_id='r', route_name='Route', direction_id='0')])[0]
        self.assertIsNone(summary['point_on_time_rate'])

    def test_custom_thresholds(self):
        row = compare_events([self.event(180)], self.static, early=0, late=120)[0]
        self.assertEqual(row['punctuality_status'], 'late')
        with self.assertRaises(ValueError):
            compare_events([], self.static, late=float('nan'))


if __name__ == '__main__':
    unittest.main()
