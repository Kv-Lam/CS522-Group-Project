"""Collect and replay a conservative, vehicle-scoped KAT arrival experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import numpy as np
from google.transit import gtfs_realtime_pb2 as pb
from punctuality import generate as generate_punctuality
from arrival_rules import crossing_events

URLS = {'static': 'https://knoxville.syncromatics.com/GTFS',
        'vehicles': 'https://knoxville.syncromatics.com/GTFS-rt/VehiclePositions',
        'trips': 'https://knoxville.syncromatics.com/GTFS-rt/TripUpdates'}
MAX_AGE = 120
MAX_GAP = 75
CLOCK_TOLERANCE = 15


# Convert a Unix timestamp to an ISO-formatted UTC string.
def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


# Write a JSON file atomically so interrupted writes do not corrupt it.
def save_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(path)


# Append and flush one JSON record to a newline-delimited journal.
def append_json(path, obj):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj) + '\n')
        stream.flush()


# Load journal records while ignoring an incomplete or invalid final line.
def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logging.warning('Ignoring an incomplete journal line in %s', path)
    return rows


# Calculate the great-circle distance between two coordinates in meters.
def distance(lat1, lon1, lat2, lon2):
    a, b = math.radians(lat1), math.radians(lat2)
    v = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(lon2-lon1)/2)**2
    return 6371000 * 2 * math.asin(min(1, math.sqrt(v)))


# Convert latitude and longitude to the local metric plane used for shapes.
def xy(lat, lon):
    return np.array([float(lon)*111195*math.cos(math.radians(36)), float(lat)*111195])


class Static:
    """Read GTFS and derive metric distances from shape coordinates."""
    # Load static GTFS tables and prepare route shape geometry.
    def __init__(self, path):
        with zipfile.ZipFile(path) as archive:
            # Read a named CSV table directly from the GTFS archive.
            def table(name):
                return list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))
            self.routes = {r['route_id']: r for r in table('routes.txt')}
            self.trips = {r['trip_id']: r for r in table('trips.txt')}
            self.stops = {r['stop_id']: r for r in table('stops.txt')}
            self.zone = ZoneInfo(table('agency.txt')[0]['agency_timezone'])
            self.frequency_trips = {r['trip_id'] for r in table('frequencies.txt')} if 'frequencies.txt' in archive.namelist() else set()
            self.times = defaultdict(list)
            for row in table('stop_times.txt'):
                row['seq'] = int(row['stop_sequence'])
                self.times[row['trip_id']].append(row)
            for rows in self.times.values():
                rows.sort(key=lambda r: r['seq'])
            shapes = defaultdict(list)
            for row in table('shapes.txt'):
                shapes[row['shape_id']].append(row)
        self.shapes = {}
        for key, rows in shapes.items():
            rows.sort(key=lambda r: int(r['shape_pt_sequence']))
            points = np.array([xy(r['shape_pt_lat'], r['shape_pt_lon']) for r in rows])
            if len(points) < 2:
                continue
            vectors = np.diff(points, axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            self.shapes[key] = (points, vectors, lengths, np.r_[0, np.cumsum(lengths)])
        self.cache = {}

    # Project a coordinate onto a shape and report its progress and ambiguity.
    def project(self, shape_id, lat, lon, lower=None, upper=None):
        points, vectors, lengths, cumulative = self.shapes[shape_id]
        fractions = np.clip(np.sum((xy(lat, lon)-points[:-1])*vectors, axis=1) /
                            np.maximum(lengths**2, 1e-9), 0, 1)
        offsets = np.linalg.norm(points[:-1]+fractions[:, None]*vectors-xy(lat, lon), axis=1)
        progress = cumulative[:-1]+fractions*lengths
        if lower is not None:
            offsets[progress < lower] = np.inf
        if upper is not None:
            offsets[progress > upper] = np.inf
        idx = int(np.argmin(offsets))
        if not np.isfinite(offsets[idx]):
            raise ValueError('No plausible shape projection')
        # Reject close competing branches that imply substantially different progress.
        ambiguous = np.any((offsets <= offsets[idx]+12) & (np.abs(progress-progress[idx]) > 200))
        return float(progress[idx]), float(offsets[idx]), bool(ambiguous)

    # Return a validated, ordered stop pattern with along-shape positions.
    def pattern(self, trip_id):
        if trip_id in self.cache:
            return self.cache[trip_id]
        trip = self.trips[trip_id]
        rows = self.times[trip_id]
        if len(rows) < 4 or trip.get('shape_id') not in self.shapes or trip.get('direction_id') not in ('0', '1'):
            raise ValueError('Missing usable shape, stops, or direction')
        ids = [r['stop_id'] for r in rows]
        first, last = self.stops[ids[0]], self.stops[ids[-1]]
        name = self.route_name(trip['route_id']).lower()
        if len(set(ids)) != len(ids) or 'loop' in name or 'circular' in name or distance(
                float(first['stop_lat']), float(first['stop_lon']), float(last['stop_lat']), float(last['stop_lon'])) < 500:
            raise ValueError('Loop or ambiguous repeated-stop pattern')
        result, previous = [], -1.0
        for row in rows:
            stop = self.stops[row['stop_id']]
            progress, offset, ambiguous = self.project(trip['shape_id'], stop['stop_lat'], stop['stop_lon'])
            if offset > 80 or ambiguous or progress < previous-10:
                raise ValueError('Unreliable stop-to-shape association')
            result.append(dict(row, along_m=progress, lat=float(stop['stop_lat']), lon=float(stop['stop_lon'])))
            previous = progress
        self.cache[trip_id] = result
        return result

    # Build the display name for a GTFS route.
    def route_name(self, route_id):
        r = self.routes[route_id]
        return (r.get('route_short_name', '')+' '+r.get('route_long_name', '')).strip()


# Deserialize a GTFS-Realtime payload and validate its header and incrementality.
def parse_feed(data):
    feed = pb.FeedMessage.FromString(data)
    if not feed.IsInitialized() or not feed.header.gtfs_realtime_version:
        raise ValueError('Invalid GTFS-Realtime header')
    if feed.header.incrementality != pb.FeedHeader.FULL_DATASET:
        raise ValueError('Differential feed is unsupported; raw bytes retained')
    return feed


# Fetch one feed, persist each response, and journal request outcomes.
def fetch(name, run, stem, timeout=15, retries=1):
    for attempt in range(retries+1):
        record = {'endpoint': name, 'request_utc': utc(time.time()), 'attempt': attempt+1, 'ok': False}
        started = time.monotonic()
        try:
            request = urllib.request.Request(URLS[name], headers={'User-Agent': 'KAT-Arrival-Research/1.0'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
                record.update(http_status=response.status, content_type=response.headers.get('Content-Type'))
            received = time.time()
            relative = f'raw/{stem}_{name}_a{attempt}.' + ('zip' if name == 'static' else 'pb')
            (run / relative).write_bytes(data)
            record.update(path=relative, received=received, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if name == 'static':
                Static(run / relative)
            else:
                feed = parse_feed(data)
                record.update(feed_timestamp=int(feed.header.timestamp), entity_count=len(feed.entity))
            record['ok'] = True
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            logging.warning('%s attempt %s failed: %s', name, attempt+1, record['error'])
        record['elapsed_seconds'] = time.monotonic()-started
        append_json(run / 'requests.jsonl', record)
        if record['ok']:
            return record
        if attempt < retries:
            time.sleep(1)
    return record


# Build a stable key for a scheduled trip instance.
def instance(descriptor):
    if not descriptor.trip_id or not descriptor.start_date or not descriptor.start_time:
        raise ValueError('Missing trip instance date or start time')
    if descriptor.schedule_relationship != pb.TripDescriptor.SCHEDULED:
        raise ValueError('Unsupported modified or unscheduled trip')
    datetime.strptime(descriptor.start_date, '%Y%m%d')
    return '|'.join([descriptor.trip_id, descriptor.start_date, descriptor.start_time])


class Processor:
    # Initialize the state used to select vehicles and replay observations.
    def __init__(self, static, label_mode='enhanced'):
        self.static = static
        self.label_mode = label_mode
        self.selected = []
        self.probes = {}
        self.history = defaultdict(list)
        self.samples = []
        self.excluded = []
        self.events = {}
        self.last = {}
        self.last_instance = {}
        self.segments = Counter()
        self.candidate_reasons = Counter()
        self.last_feed = {}
        self.feed_issues = Counter()

    # Record an observation or sample that failed a quality rule.
    def reject(self, reason, **record):
        self.excluded.append(dict(record, reason=reason))

    # Validate a realtime vehicle entity and normalize its observation fields.
    def observation(self, vehicle, receipt):
        key = instance(vehicle.trip)
        trip = self.static.trips[vehicle.trip.trip_id]
        if vehicle.trip.route_id and vehicle.trip.route_id != trip['route_id']:
            raise ValueError('Route mismatch')
        if vehicle.trip.HasField('direction_id') and str(vehicle.trip.direction_id) != trip['direction_id']:
            raise ValueError('Direction mismatch')
        pattern = self.static.pattern(vehicle.trip.trip_id)
        if not vehicle.vehicle.id or not vehicle.HasField('position') or not vehicle.HasField('timestamp'):
            raise ValueError('Missing vehicle ID, position, or observation timestamp')
        if not -CLOCK_TOLERANCE <= receipt-vehicle.timestamp <= MAX_AGE:
            raise ValueError('Stale or future vehicle observation')
        lat, lon = vehicle.position.latitude, vehicle.position.longitude
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            raise ValueError('Invalid coordinates')
        lower = upper = None
        seq = int(vehicle.current_stop_sequence) if vehicle.HasField('current_stop_sequence') else None
        if seq is not None:
            indices = [i for i, s in enumerate(pattern) if s['seq'] == seq]
            if not indices:
                raise ValueError('Stop sequence mismatch')
            i = indices[0]
            if vehicle.stop_id and vehicle.stop_id != pattern[i]['stop_id']:
                raise ValueError('Stop ID mismatch')
            lower, upper = pattern[max(0, i-1)]['along_m']-150, pattern[i]['along_m']+150
        progress, offset, ambiguous = self.static.project(trip['shape_id'], lat, lon, lower, upper)
        if offset > 80 or ambiguous:
            raise ValueError('Off-route or ambiguous GPS projection')
        return {'vehicle_id': vehicle.vehicle.id, 'trip_id': vehicle.trip.trip_id,
                'trip_instance_id': key, 'service_date': vehicle.trip.start_date,
                'route_id': trip['route_id'], 'route_name': self.static.route_name(trip['route_id']),
                'direction_id': trip['direction_id'], 'gps_timestamp': int(vehicle.timestamp),
                'observation_timestamp': receipt, 'observation_utc': utc(receipt),
                'observation_local': datetime.fromtimestamp(receipt, self.static.zone).isoformat(),
                'latitude': lat, 'longitude': lon, 'along_position_m': progress,
                'shape_offset_m': offset, 'current_stop_sequence': seq,
                'vehicle_status': int(vehicle.current_status) if vehicle.HasField('current_status') else None}

    # Load a feed only when its request, timestamp, and ordering are valid.
    def accept_feed(self, kind, record, run):
        if not record or not record.get('ok'):
            self.feed_issues[kind+'_request_failed'] += 1
            return None
        feed = parse_feed((run / record['path']).read_bytes())
        stamp = int(feed.header.timestamp)
        if not stamp or not -CLOCK_TOLERANCE <= record['received']-stamp <= MAX_AGE:
            self.feed_issues[kind+'_stale_or_missing_timestamp'] += 1
            return None
        if stamp <= self.last_feed.get(kind, 0):
            self.feed_issues[kind+'_duplicate_or_out_of_order'] += 1
            return None
        self.last_feed[kind] = stamp
        return feed

    # Discover moving vehicles from each feed snapshot without dropping prior selections.
    def discover(self, observations):
        groups = defaultdict(list)
        selected_keys = {(s['route_id'], s['direction_id'], s['vehicle_id']) for s in self.selected}
        selected_vehicles = {s['vehicle_id'] for s in self.selected}
        active_selected = Counter()
        active_directions = {}
        for obs in observations:
            key = (obs['route_id'], obs['direction_id'], obs['vehicle_id'])
            if key in selected_keys:
                active_selected[(obs['route_id'], obs['direction_id'])] += 1
                active_directions[obs['route_id']] = obs['direction_id']
            previous = self.probes.get(obs['vehicle_id'])
            if previous and previous['trip_instance_id'] == obs['trip_instance_id']:
                dt = obs['gps_timestamp']-previous['gps_timestamp']
                movement = obs['along_position_m']-previous['along_position_m']
                if 0 < dt <= MAX_GAP and 10 < movement < 35*dt:
                    groups[(obs['route_id'], obs['direction_id'])].append(obs)
            self.probes[obs['vehicle_id']] = obs

        chosen_routes = set(active_directions)
        for (route, direction), group in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            if route in active_directions and active_directions[route] != direction:
                continue
            if route not in chosen_routes:
                if len(chosen_routes) == 2:
                    continue
                chosen_routes.add(route)
            slots = max(0, 2-active_selected[(route, direction)])
            candidates = [obs for obs in sorted(group, key=lambda o: o['vehicle_id'])
                          if obs['vehicle_id'] not in selected_vehicles]
            for obs in candidates[:slots]:
                self.selected.append({k: obs[k] for k in ('route_id', 'route_name', 'direction_id', 'vehicle_id')})
                selected_vehicles.add(obs['vehicle_id'])
                active_selected[(route, direction)] += 1
                logging.info('Selected route=%s direction=%s vehicle=%s', obs['route_name'], direction, obs['vehicle_id'])
            if len(chosen_routes) == 2:
                break

    # Pair feed snapshots, select eligible vehicles, and process observations.
    def consume(self, tick, run):
        vf = self.accept_feed('vehicles', tick.get('vehicles'), run)
        tf = self.accept_feed('trips', tick.get('trips'), run)
        if vf is None:
            return
        receipt = tick['vehicles']['received']
        # Features become available only after both requests have completed.
        available = max(receipt, tick.get('trips', {}).get('received', receipt))
        updates = {}
        if tf is not None:
            for entity in tf.entity:
                if not entity.HasField('trip_update') or entity.is_deleted:
                    continue
                update = entity.trip_update
                try:
                    key = instance(update.trip)
                    stamp = update.timestamp or tf.header.timestamp
                    if not -CLOCK_TOLERANCE <= available-stamp <= MAX_AGE:
                        continue
                    updates[(update.vehicle.id, key)] = update
                except ValueError:
                    continue
        observations = []
        seen = set()
        for entity in vf.entity:
            if not entity.HasField('vehicle') or entity.is_deleted:
                continue
            vehicle = entity.vehicle
            try:
                obs = self.observation(vehicle, available)
                if obs['vehicle_id'] in seen:
                    raise ValueError('Duplicate vehicle entity')
                seen.add(obs['vehicle_id'])
                observations.append(obs)
            except (ValueError, KeyError) as exc:
                self.candidate_reasons[str(exc)] += 1
                if any(s['vehicle_id'] == vehicle.vehicle.id for s in self.selected):
                    self.reject(str(exc), record_type='observation', vehicle_id=vehicle.vehicle.id,
                                observation_utc=utc(available))
        self.discover(observations)
        for obs in observations:
            if not any(all(obs[k] == s[k] for k in ('route_id', 'direction_id', 'vehicle_id')) for s in self.selected):
                continue
            self.process_observation(obs, updates.get((obs['vehicle_id'], obs['trip_instance_id'])))

    # Track vehicle continuity, infer arrivals, and create downstream samples.
    def process_observation(self, obs, update):
        vid, inst = obs['vehicle_id'], obs['trip_instance_id']
        key = (vid, inst)
        prev = self.last.get(vid)
        if prev and obs['gps_timestamp'] <= prev['gps_timestamp']:
            self.reject('Duplicate or out-of-order vehicle timestamp', record_type='observation', **obs)
            return
        discontinuity = prev and (prev['trip_instance_id'] != inst or
                                  obs['gps_timestamp']-prev['gps_timestamp'] > MAX_GAP)
        if prev and prev['trip_instance_id'] == inst:
            dt = obs['gps_timestamp']-prev['gps_timestamp']
            delta = obs['along_position_m']-prev['along_position_m']
            if delta < -60 or delta > 35*dt:
                self.reject('Implausible progress or backwards movement', record_type='observation', **obs)
                self.segments[key] += 1
                self.last[vid] = obs
                return
        if discontinuity:
            self.segments[key] += 1
        obs['continuity_id'] = f'{vid}|{inst}|{self.segments[key]}'
        history = self.history[obs['continuity_id']]
        history.append(obs)
        self.last[vid] = obs
        pattern = self.static.pattern(obs['trip_id'])
        if len(history) >= 3:
            self.detect_arrivals(history[-3:], pattern)
            if self.label_mode == 'enhanced':
                for event in crossing_events(history[-3:], pattern, CLOCK_TOLERANCE):
                    event['arrival_utc'] = utc(event['arrival_timestamp'])
                    self.events.setdefault(event['arrival_event_id'], event)
        # Only create features from the current snapshot, never from later observations.
        downstream = [s for s in pattern if s['along_m'] > obs['along_position_m']+20][:3]
        for stop in downstream:
            eta, eta_uncertainty = None, None
            skipped = False
            if update is not None:
                for stu in update.stop_time_update:
                    if stu.HasField('stop_sequence') and stu.stop_sequence == stop['seq'] and (not stu.stop_id or stu.stop_id == stop['stop_id']):
                        skipped = stu.schedule_relationship == pb.TripUpdate.StopTimeUpdate.SKIPPED
                        if stu.schedule_relationship == pb.TripUpdate.StopTimeUpdate.SCHEDULED and stu.HasField('arrival') and stu.arrival.HasField('time'):
                            eta = int(stu.arrival.time)
                            eta_uncertainty = stu.arrival.uncertainty if stu.arrival.HasField('uncertainty') else None
                        break
            sample = dict(obs, sample_id=f'{vid}|{inst}|{obs["gps_timestamp"]}|{stop["seq"]}',
                          target_stop_id=stop['stop_id'], target_stop_sequence=stop['seq'],
                          straight_line_distance_m=distance(obs['latitude'], obs['longitude'], stop['lat'], stop['lon']),
                          along_route_distance_m=stop['along_m']-obs['along_position_m'],
                          distance_method='GTFS shape projection; local metric approximation',
                          official_eta_timestamp=eta,
                          official_eta_remaining_seconds=eta-obs['observation_timestamp'] if eta else None,
                          official_eta_uncertainty_seconds=eta_uncertainty,
                          official_eta_status='skipped' if skipped else ('available' if eta else 'missing'),
                          gps_age_seconds=obs['observation_timestamp']-obs['gps_timestamp'])
            self.samples.append(sample)

    # Infer stop-passage events from a three-point GPS observation window.
    def detect_arrivals(self, points, pattern):
        a, b, c = points
        if not (0 < b['gps_timestamp']-a['gps_timestamp'] <= MAX_GAP and
                0 < c['gps_timestamp']-b['gps_timestamp'] <= MAX_GAP):
            return
        for stop in pattern:
            progress = stop['along_m']
            distances = [distance(p['latitude'], p['longitude'], stop['lat'], stop['lon']) for p in points]
            # Require approach, a near-stop observation, and subsequent departure.
            if not (a['along_position_m'] < progress-35 and abs(b['along_position_m']-progress) <= 45
                    and c['along_position_m'] > progress+35 and distances[1] <= 45
                    and distances[0] > distances[1]+15 and distances[2] > distances[1]+15):
                continue
            if any(p['current_stop_sequence'] is not None and
                   ((p is a and p['current_stop_sequence'] > stop['seq']) or
                    (p is c and p['current_stop_sequence'] < stop['seq'])) for p in points):
                continue
            event_id = f'{b["vehicle_id"]}|{b["trip_instance_id"]}|{stop["seq"]}'
            if event_id in self.events and self.events[event_id]['label_confidence'] != 'low_crossing':
                continue
            status_support = b['vehicle_status'] == pb.VehiclePosition.STOPPED_AT and b['current_stop_sequence'] == stop['seq']
            lower, upper = a['gps_timestamp']-CLOCK_TOLERANCE, c['gps_timestamp']+CLOCK_TOLERANCE
            self.events[event_id] = {
                'arrival_event_id': event_id, 'trip_instance_id': b['trip_instance_id'],
                'continuity_id': b['continuity_id'], 'vehicle_id': b['vehicle_id'],
                'route_id': b['route_id'], 'route_name': b['route_name'], 'direction_id': b['direction_id'],
                'service_date': b['service_date'], 'target_stop_id': stop['stop_id'], 'target_stop_sequence': stop['seq'],
                'arrival_timestamp': b['gps_timestamp'], 'arrival_utc': utc(b['gps_timestamp']),
                'arrival_lower_timestamp': lower, 'arrival_upper_timestamp': upper,
                'time_uncertainty_seconds': max(b['gps_timestamp']-lower, upper-b['gps_timestamp']),
                'label_source': 'gps_inferred', 'label_confidence': 'medium_status_supported' if status_support else 'medium',
                'label_rule': 'three_point_approach_near_depart_v1',
                'rule_note': 'Stop passage inferred; passenger boarding or door opening is not confirmed'}


SAMPLE_FIELDS = ['sample_id', 'trip_instance_id', 'continuity_id', 'service_date', 'route_id', 'route_name',
                 'direction_id', 'vehicle_id', 'target_stop_id', 'target_stop_sequence', 'observation_timestamp',
                 'observation_utc', 'observation_local', 'gps_timestamp', 'gps_age_seconds', 'latitude', 'longitude',
                 'straight_line_distance_m', 'along_route_distance_m', 'official_eta_timestamp',
                 'official_eta_remaining_seconds', 'arrival_event_id', 'remaining_travel_time_seconds',
                 'label_source', 'label_confidence', 'time_uncertainty_seconds', 'quality_flags']
EVENT_FIELDS = ['arrival_event_id', 'trip_instance_id', 'vehicle_id', 'target_stop_id', 'arrival_timestamp',
                'arrival_lower_timestamp', 'arrival_upper_timestamp', 'label_source', 'label_confidence', 'time_uncertainty_seconds']


# Write CSV rows using the requested columns plus any additional row fields.
def write_csv(path, rows, fields):
    fields = list(dict.fromkeys(fields+[k for row in rows for k in row]))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Label samples and generate the run's data files, reports, evaluations, and plots.
def outputs(run, processor, requests, ticks, error=None, early=60, late=300):
    p = processor
    events = list(p.events.values()) if p else []
    labeled, excluded = [], list(p.excluded) if p else []
    for sample in p.samples if p else []:
        event_id = f'{sample["vehicle_id"]}|{sample["trip_instance_id"]}|{sample["target_stop_sequence"]}'
        event = p.events.get(event_id)
        reason = None
        if sample['official_eta_status'] == 'skipped':
            reason = 'Target explicitly skipped in the observation snapshot'
        elif event is None:
            last = p.last.get(sample['vehicle_id'], {})
            if last.get('trip_instance_id') != sample['trip_instance_id']:
                reason = 'Trip changed before a reliable arrival'
            elif last.get('continuity_id') != sample['continuity_id']:
                reason = 'Data gap or discontinuity before a reliable arrival'
            else:
                reason = 'No reliable arrival observed before collection ended'
        elif event['continuity_id'] != sample['continuity_id']:
            reason = 'Data gap or discontinuity between sample and arrival'
        elif sample['observation_timestamp'] >= event['arrival_lower_timestamp']:
            reason = 'Observation overlaps the arrival uncertainty interval'
        if reason:
            excluded.append(dict(sample, record_type='sample', reason=reason))
        else:
            labeled.append(dict(sample, arrival_event_id=event_id,
                                remaining_travel_time_seconds=event['arrival_timestamp']-sample['observation_timestamp'],
                                remaining_time_lower_seconds=event['arrival_lower_timestamp']-sample['observation_timestamp'],
                                remaining_time_upper_seconds=event['arrival_upper_timestamp']-sample['observation_timestamp'],
                                label_source=event['label_source'], label_confidence=event['label_confidence'],
                                label_rule=event['label_rule'],
                                time_uncertainty_seconds=event['time_uncertainty_seconds'],
                                quality_flags='inferred_stop_passage;interval_censored;correlated_event_samples'))
    write_csv(run / 'arrival_events.csv', events, EVENT_FIELDS)
    write_csv(run / 'training_samples.csv', labeled, SAMPLE_FIELDS)
    write_csv(run / 'excluded_records.csv', excluded, ['record_type', 'reason']+SAMPLE_FIELDS)
    write_csv(run / 'observations.csv', [o for h in p.history.values() for o in h] if p else [],
              ['vehicle_id', 'trip_instance_id', 'continuity_id', 'gps_timestamp', 'observation_utc'])
    selected = p.selected if p else []
    save_json(run / 'selection.json', selected)
    lines = ['# KAT arrival experiment report', '', f'Output directory: `{run}`',
             f'Collection ticks: {len(ticks)}', f'Fatal collection/replay error: {error or "None"}', '',
             '## Source availability and updates', '',
             '| Feed | Successful attempts / all attempts | Successful ticks | Unique payloads | Unique feed timestamps |',
             '|---|---:|---:|---:|---:|']
    for name in URLS:
        rows = [r for r in requests if r['endpoint'] == name]
        good = [r for r in rows if r['ok']]
        success_ticks = sum(bool(t.get(name, {}).get('ok')) for t in ticks)
        rate = f'{len(good)}/{len(rows)} ({len(good)/len(rows):.1%})' if rows else '0/0'
        lines.append(f'| {name} | {rate} | {success_ticks if name != "static" else "N/A"} | {len({r["sha256"] for r in good})} | {len({r.get("feed_timestamp") for r in good if r.get("feed_timestamp")})} |')
    lines += ['', 'Changes in payload/header timestamps do not alone prove vehicle movement. Selection requires forward movement across two fresh observations.',
              '', '## Coverage', '']
    for s in selected:
        observations = [o for h in p.history.values() for o in h if o['vehicle_id'] == s['vehicle_id']]
        stamps = sorted(o['gps_timestamp'] for o in observations)
        gaps = [b-a for a,b in zip(stamps, stamps[1:])]
        lines.append(f'- {s["route_name"]}; direction {s["direction_id"]}; vehicle {s["vehicle_id"]}: {len(stamps)} accepted observations; maximum gap {max(gaps, default=0)} s; {len({o["trip_instance_id"] for o in observations})} trip instances.')
    if not selected:
        lines.append('No qualifying moving, non-loop route/vehicle pairs were verified. No route numbers were guessed.')
    lines += ['', f'Candidate validation failures (network selection audit only): {dict(p.candidate_reasons) if p else {}}',
              f'Feed quality failures: {dict(p.feed_issues) if p else {}}', '', '## Labels and dataset', '',
              f'- Independent arrival events: {len(events)}', f'- Valid labeled samples: {len(labeled)}',
              f'- Unlabeled/excluded records: {len(excluded)}',
              f'- Label quality: {dict(Counter(e["label_confidence"] for e in events))}',
              f'- Exclusion reasons: {dict(Counter(e["reason"] for e in excluded))}',
              f'- Missing official ETA among candidate samples: {sum(s["official_eta_timestamp"] is None for s in p.samples) if p else 0}',
              '', '## Official ETA error', '',
              'Error = official predicted arrival epoch minus label arrival epoch; positive means the prediction was late. Repeated samples share arrival labels.']
    for source in ['direct_observed', 'gps_inferred']:
        matches = [s for s in labeled if s['label_source'] == source and s['official_eta_timestamp'] is not None]
        unique = {s['arrival_event_id'] for s in matches}
        if len(matches) >= 5 and len(unique) >= 3:
            errors = np.array([s['official_eta_timestamp']-p.events[s['arrival_event_id']]['arrival_timestamp'] for s in matches])
            grouped = defaultdict(list)
            for s, err in zip(matches, errors):
                grouped[s['arrival_event_id']].append(abs(float(err)))
            lines.append(f'- {source}: n={len(matches)}, events={len(unique)}, bias={errors.mean():.1f} s, MAE={np.abs(errors).mean():.1f} s, RMSE={np.sqrt(np.mean(errors**2)):.1f} s; event-balanced MAE={np.mean([np.mean(v) for v in grouped.values()]):.1f} s.')
        else:
            lines.append(f'- {source}: insufficient matches ({len(matches)} samples, {len(unique)} events); requires at least 5 samples and 3 events.')
    lines += ['', '## Rules, limitations and next steps', '',
              '- TripUpdates arrival times are predictions, including past times or uncertainty=0. This feed provides no verified actual-arrival flag, so no direct-observed labels are asserted.',
              '- Strict GPS labels require approach, a point within 45 m, and departure. Enhanced labels additionally bracket a crossing with two points and confirm forward motion with a third. Both estimate stop passage, not door opening.',
              '- Strict arrival estimates use the middle GPS time; crossing estimates interpolate shape progress inside the bracketing interval. Bounds include clock tolerance. Confidence is a rule category, not a calibrated probability.',
              '- Source clock lead of up to 15 seconds is tolerated; arrival bounds are widened by 15 seconds on each side. Negative GPS age records are retained explicitly; the system clock is never modified.',
              '- Freshness limit is 120 s; maximum continuity gap is 75 s. Ambiguous projections, repeated stops, near-closed patterns and implausible motion are excluded.',
              '- All analysis is restricted to selected vehicle IDs, routes and directions. Selection is frozen once qualifying routes are found; vehicle changes to other routes/directions are not followed.',
              '- Features use receipt-time availability and GPS age. No future ETA, position, or target list is backfilled. Only the next three downstream stops are sampled.',
              '- UTC epoch arithmetic handles midnight and DST. Trip identity includes the explicit GTFS service date and start time (including hours beyond 24); no service date is guessed.',
              '- Along-route distances are calculated from shape geometry in approximate meters; straight-line distances use haversine. Producer shape-distance units are not assumed.',
              '- Raw full-network feeds are retained for replay. Candidate rejection counts describe selection only; observations, samples and events concern selected vehicles only.',
              '- Missing a stop between 30-second snapshots can leave it unlabeled. A short or off-hours run can produce no events. End-of-run censoring is never filled with predictions.',
              '- Split future datasets by service date or trip instance and keep an arrival event in one split. Collect more independent events before assessing ETA accuracy or training.',
              '', 'Specification: https://gtfs.org/documentation/realtime/reference/']
    lines += generate_punctuality(run, events, p.static if p else None, selected, early, late)
    from evaluate import evaluate_runs
    evaluation = evaluate_runs([run], run / 'evaluation')
    lines += ['', '## Matched model comparison', '', evaluation,
              'See evaluation/report.md, predictions.csv and model_comparison.png. All methods use identical held-out samples.']
    (run / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in selected:
        observations = sorted([o for h in p.history.values() for o in h if o['vehicle_id'] == s['vehicle_id']], key=lambda o: o['gps_timestamp'])
        if observations:
            ax.scatter([(o['gps_timestamp']-observations[0]['gps_timestamp'])/60 for o in observations],
                       [o['along_position_m']/1000 for o in observations], s=12, label=s['route_name']+' / '+s['vehicle_id'])
    ax.set(xlabel='Minutes from first accepted GPS observation', ylabel='Progress along trip shape (km)', title='Selected vehicle coverage')
    if selected:
        ax.legend(fontsize=7)
    else:
        ax.text(.5, .5, 'No qualifying vehicle observations', transform=ax.transAxes, ha='center')
    fig.tight_layout(); fig.savefig(run / 'vehicle_progress.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(['Arrival events', 'Labeled samples', 'Unlabeled samples'],
           [len(events), len(labeled), sum(r.get('record_type') == 'sample' for r in excluded)])
    ax.set(title='Dataset availability', ylabel='Count')
    fig.tight_layout(); fig.savefig(run / 'dataset_quality.png', dpi=140); plt.close(fig)
    logging.info('Report saved: %s; events=%d labeled_samples=%d', run / 'report.md', len(events), len(labeled))


# Replay saved feeds and regenerate derived outputs without making network calls.
def postprocess(run, early=60, late=300, label_mode='enhanced'):
    save_json(run / 'processing_config.json', {'rule_version': 'arrival_rules_v2', 'label_mode': label_mode,
              'max_age_seconds': MAX_AGE, 'max_gap_seconds': MAX_GAP,
              'clock_tolerance_seconds': CLOCK_TOLERANCE,
              'punctuality_early_seconds': early, 'punctuality_late_seconds': late,
              'punctuality_rule_version': 'schedule_arrival_interval_v1'})
    requests, ticks = read_jsonl(run / 'requests.jsonl'), read_jsonl(run / 'ticks.jsonl')
    static_record = next((r for r in requests if r['endpoint'] == 'static' and r['ok']), None)
    processor, error = None, None
    try:
        if static_record:
            processor = Processor(Static(run / static_record['path']), label_mode)
            for tick in ticks:
                processor.consume(tick, run)
        else:
            error = 'Static GTFS unavailable or invalid; no reliable route association is possible'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        logging.exception('Replay failed')
    outputs(run, processor, requests, ticks, error, early, late)
    return error


# Collect live feeds for the requested duration, then save and postprocess them.
def collect(args):
    run = Path(args.output).resolve() if args.output else ROOT / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'raw').mkdir()
    handler = logging.FileHandler(run / 'collection.log', encoding='utf-8')
    logging.getLogger().addHandler(handler)
    metadata = {'started_utc': utc(time.time()), 'duration_seconds': args.duration_minutes*60,
                'interval_seconds': args.interval, 'python': sys.executable, 'urls': URLS,
                'punctuality_early_seconds': getattr(args, 'early_seconds', 60),
                'punctuality_late_seconds': getattr(args, 'late_seconds', 300),
                'max_age_seconds': MAX_AGE, 'max_gap_seconds': MAX_GAP, 'clock_tolerance_seconds': CLOCK_TOLERANCE, 'status': 'running'}
    save_json(run / 'run.json', metadata)
    logging.info('Output directory: %s', run)
    try:
        static_record = fetch('static', run, 'static', args.timeout, args.retries)
        processor = Processor(Static(run / static_record['path'])) if static_record.get('ok') else None
        start = time.monotonic()
        deadline = start+args.duration_minutes*60
        index = 0
        while time.monotonic() < deadline:
            tick = {'index': index}
            # Fetch TripUpdates first so the paired prediction existed before the GPS receipt.
            for name in ['trips', 'vehicles']:
                tick[name] = fetch(name, run, f'{index:06d}', args.timeout, args.retries)
            append_json(run / 'ticks.jsonl', tick)
            if processor:
                processor.consume(tick, run)
                save_json(run / 'selection.json', processor.selected)
            logging.info('Progress %.0f/%.0f seconds; tick=%d; vehicles_ok=%s trips_ok=%s selected=%d',
                         time.monotonic()-start, args.duration_minutes*60, index,
                         tick['vehicles']['ok'], tick['trips']['ok'], len(processor.selected) if processor else 0)
            index += 1
            next_tick = start+index*args.interval
            while time.monotonic() < min(next_tick, deadline):
                time.sleep(max(0, min(.5, min(next_tick, deadline)-time.monotonic())))
            if time.monotonic() > next_tick+args.interval:
                index = math.floor((time.monotonic()-start)/args.interval)+1
        metadata['status'] = 'completed'
    except KeyboardInterrupt:
        logging.info('Interrupt received; saving and postprocessing collected evidence')
        metadata['status'] = 'interrupted'
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        logging.exception('Collection failed; postprocessing available evidence')
    finally:
        metadata['ended_utc'] = utc(time.time())
        save_json(run / 'run.json', metadata)
        error = postprocess(run, early=getattr(args, 'early_seconds', 60), late=getattr(args, 'late_seconds', 300))
        logging.getLogger().removeHandler(handler)
        handler.close()
    return 1 if metadata['status'] == 'failed' or error else 0


# Define the collect and postprocess commands and their command-line options.
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect_parser = commands.add_parser('collect')
    collect_parser.add_argument('--duration-minutes', type=float, default=60)
    collect_parser.add_argument('--interval', type=float, default=30)
    collect_parser.add_argument('--timeout', type=float, default=15)
    collect_parser.add_argument('--retries', type=int, default=1)
    collect_parser.add_argument('--output', help='New output directory; must not exist')
    replay_parser = commands.add_parser('postprocess')
    replay_parser.add_argument('run_directory', type=Path)
    replay_parser.add_argument('--label-mode', choices=['strict', 'enhanced'], default='enhanced')
    for command_parser in (collect_parser, replay_parser):
        command_parser.add_argument('--early-seconds', type=float, default=60,
                                    help='Allowed early arrival for research punctuality (default: 60)')
        command_parser.add_argument('--late-seconds', type=float, default=300,
                                    help='Allowed late arrival for research punctuality (default: 300)')
    return parser


# Parse command-line arguments, defaulting to a collection run when omitted.
def parse_args(parser):
    return parser.parse_args(sys.argv[1:] or ['collect'])


# Set the default format and level for application log messages.
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# Reject invalid punctuality tolerances and collection settings.
def validate_args(parser, args):
    if not all(math.isfinite(v) and v >= 0 for v in (args.early_seconds, args.late_seconds)):
        parser.error('Punctuality tolerances must be finite and nonnegative')
    if args.command == 'collect' and (
            args.duration_minutes <= 0 or args.interval <= 0 or args.timeout <= 0 or not 0 <= args.retries <= 3):
        parser.error('Duration, interval and timeout must be positive; retries must be between 0 and 3')


# Dispatch the selected CLI command and return its process exit status.
def run_command(args):
    if args.command == 'postprocess':
        error = postprocess(args.run_directory.resolve(), args.early_seconds, args.late_seconds, args.label_mode)
        return 1 if error else 0
    return collect(args)


# Run the CLI workflow in order:
# 1. Build the command parser and parse the user's arguments.
# 2. Configure logging, then validate the parsed arguments.
# 3. Dispatch to collect() or postprocess() and return its exit status.
def main():
    parser = build_parser()
    args = parse_args(parser)
    configure_logging()
    validate_args(parser, args)
    return run_command(args)


if __name__ == '__main__':
    raise SystemExit(main())
