"""Additional interval-censored stop-passage labels from continuous GPS tracks."""
import math


def crossing_events(points, stops, clock_tolerance=15):
    """Bracket a crossing with two points and confirm forward motion with a third."""
    a, b, c = points
    if any(p.get('continuity_id') != a.get('continuity_id') or
           p['trip_instance_id'] != a['trip_instance_id'] or p['vehicle_id'] != a['vehicle_id'] for p in points):
        return []
    dt = b['gps_timestamp']-a['gps_timestamp']
    follow_dt = c['gps_timestamp']-b['gps_timestamp']
    delta = b['along_position_m']-a['along_position_m']
    follow_delta = c['along_position_m']-b['along_position_m']
    if not (0 < dt <= 60 and 0 < follow_dt <= 60 and 50 <= delta <= 600 and
            delta/dt <= 22 and 5 <= follow_delta <= min(600, 22*follow_dt)):
        return []
    if any(p.get('shape_offset_m', float('inf')) > 35 for p in points):
        return []
    def xy(p):
        return (p['longitude']*111195*math.cos(math.radians(36)), p['latitude']*111195)
    ax, ay = xy(a)
    bx, by = xy(b)
    vx, vy = bx-ax, by-ay
    chord = math.hypot(vx, vy)
    if not .55 <= chord/delta <= 1.5:
        return []
    events = []
    for stop in stops:
        before = stop['along_m']-a['along_position_m']
        after = b['along_position_m']-stop['along_m']
        if not (25 <= before <= 350 and 25 <= after <= 350):
            continue
        if any(p.get('current_stop_sequence') is not None and
               ((p is a and p['current_stop_sequence'] > stop['seq']) or
                (p is b and p['current_stop_sequence'] < stop['seq']) or
                (p is c and p['current_stop_sequence'] < stop['seq'])) for p in points):
            continue
        sx, sy = xy({'latitude': stop['lat'], 'longitude': stop['lon']})
        fraction = max(0, min(1, ((sx-ax)*vx+(sy-ay)*vy)/max(chord**2, 1)))
        if math.hypot(sx-(ax+fraction*vx), sy-(ay+fraction*vy)) > 60:
            continue
        estimate = a['gps_timestamp']+dt*before/delta
        lower = a['gps_timestamp']-clock_tolerance
        upper = b['gps_timestamp']+clock_tolerance
        events.append(dict(
            arrival_event_id=f'{a["vehicle_id"]}|{a["trip_instance_id"]}|{stop["seq"]}',
            **{k: a[k] for k in ('trip_instance_id', 'continuity_id', 'vehicle_id', 'route_id',
                                 'route_name', 'direction_id', 'service_date')},
            target_stop_id=stop['stop_id'], target_stop_sequence=stop['seq'],
            arrival_timestamp=estimate, arrival_lower_timestamp=lower, arrival_upper_timestamp=upper,
            time_uncertainty_seconds=max(estimate-lower, upper-estimate),
            label_source='gps_inferred', label_confidence='low_crossing',
            label_rule='bracketed_crossing_confirmed_v1',
            evidence_before_timestamp=a['gps_timestamp'], evidence_after_timestamp=b['gps_timestamp'],
            evidence_confirmation_timestamp=c['gps_timestamp'],
            evidence_before_distance_m=before, evidence_after_distance_m=after,
            rule_note='Interpolated stop passage with a GPS time bracket; stopping is not confirmed'))
    return events
