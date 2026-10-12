"""Compare unique observed arrival events with archived GTFS schedules."""
import csv
from collections import Counter
from datetime import datetime, timezone
import math

import numpy as np


def gtfs_seconds(value):
    parts = value.split(':')
    if len(parts) != 3 or any(not p.isdigit() for p in parts):
        raise ValueError('Missing or invalid GTFS time')
    hour, minute, second = map(int, parts)
    if minute >= 60 or second >= 60:
        raise ValueError('Invalid GTFS minutes or seconds')
    return hour * 3600 + minute * 60 + second


def scheduled_epoch(service_date, value, zone):
    """Use GTFS service-day noon minus 12 elapsed hours, including DST days."""
    noon = datetime.strptime(service_date, '%Y%m%d').replace(hour=12, tzinfo=zone)
    return noon.timestamp() - 12 * 3600 + gtfs_seconds(value)


def classify(delay, early, late):
    return 'early' if delay < -early else ('late' if delay > late else 'on_time')


def compare_events(events, static, early=60, late=300):
    """Return one schedule comparison per event, retaining unmatched events."""
    if not all(math.isfinite(v) and v >= 0 for v in (early, late)):
        raise ValueError('Punctuality tolerances must be finite and nonnegative')
    rows, seen = [], set()
    for event in events:
        if event['arrival_event_id'] in seen:
            continue
        seen.add(event['arrival_event_id'])
        row = dict(event, schedule_match_status='excluded', schedule_exclusion_reason='',
                   early_tolerance_seconds=early, late_tolerance_seconds=late)
        try:
            if static is None:
                raise ValueError('Static GTFS unavailable')
            trip_id, date, start = event['trip_instance_id'].split('|')
            trip = static.trips[trip_id]
            if date != event['service_date'] or trip['route_id'] != event['route_id'] or trip['direction_id'] != str(event['direction_id']):
                raise ValueError('Trip, route, direction or service date mismatch')
            if trip_id in getattr(static, 'frequency_trips', set()):
                raise ValueError('Frequency-based trip needs a separate schedule mapping')
            stops = static.times[trip_id]
            if gtfs_seconds(start) != gtfs_seconds(stops[0]['departure_time']):
                raise ValueError('Trip instance start does not match the archived schedule')
            matches = [s for s in stops if s['seq'] == int(event['target_stop_sequence'])
                       and s['stop_id'] == event['target_stop_id']]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous scheduled stop')
            stop = matches[0]
            planned = scheduled_epoch(date, stop['arrival_time'], static.zone)
            estimate = float(event['arrival_timestamp'])
            lower, upper = float(event['arrival_lower_timestamp']), float(event['arrival_upper_timestamp'])
            if not all(math.isfinite(v) for v in (estimate, lower, upper)) or not lower <= estimate <= upper:
                raise ValueError('Invalid arrival uncertainty interval')
            point = stop.get('timepoint', '') or '1'
            if point not in ('0', '1'):
                raise ValueError('Invalid schedule timepoint value')
            lo, hi, delay = lower-planned, upper-planned, estimate-planned
            status = classify(lo, early, late)
            if status != classify(hi, early, late):
                status = 'uncertain'
            row.update(schedule_match_status='matched', scheduled_arrival_time=stop['arrival_time'],
                       scheduled_arrival_timestamp=planned,
                       scheduled_arrival_utc=datetime.fromtimestamp(planned, timezone.utc).isoformat(),
                       scheduled_arrival_local=datetime.fromtimestamp(planned, static.zone).isoformat(),
                       schedule_precision='exact_timepoint' if point == '1' else 'estimated_stop_time',
                       delay_seconds=delay, delay_minutes=delay/60,
                       delay_lower_seconds=lo, delay_upper_seconds=hi,
                       punctuality_point_status=classify(delay, early, late), punctuality_status=status,
                       possibly_on_time=lo <= late and hi >= -early)
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            row['schedule_exclusion_reason'] = str(exc)
        rows.append(row)
    return rows


def summarize(rows, selected):
    """Separate routes, directions, label sources and schedule precision."""
    groups = {(r['route_id'], str(r['direction_id']), r['label_source'], r['schedule_precision'])
              for r in rows if r['schedule_match_status'] == 'matched'}
    for s in selected:
        if not any(g[:2] == (s['route_id'], str(s['direction_id'])) for g in groups):
            groups.add((s['route_id'], str(s['direction_id']), 'no_matched_events', 'not_available'))
    result = []
    for route, direction, source, precision in sorted(groups):
        subset = [r for r in rows if r['schedule_match_status'] == 'matched' and
                  (r['route_id'], str(r['direction_id']), r['label_source'], r['schedule_precision']) ==
                  (route, direction, source, precision)]
        names = [r['route_name'] for r in subset] or [s['route_name'] for s in selected if s['route_id'] == route]
        counts = Counter(r['punctuality_status'] for r in subset)
        n = len(subset)
        delays = [r['delay_seconds'] for r in subset]
        result.append(dict(route_id=route, route_name=names[0], direction_id=direction,
                           label_source=source, schedule_precision=precision, matched_events=n,
                           trip_instances=len({r['trip_instance_id'] for r in subset}),
                           early_events=counts['early'], on_time_events=counts['on_time'],
                           late_events=counts['late'], uncertain_events=counts['uncertain'],
                           point_on_time_rate=sum(r['punctuality_point_status'] == 'on_time' for r in subset)/n if n else None,
                           on_time_rate_lower=counts['on_time']/n if n else None,
                           on_time_rate_upper=sum(r['possibly_on_time'] for r in subset)/n if n else None,
                           mean_delay_seconds=float(np.mean(delays)) if n else None,
                           median_delay_seconds=float(np.median(delays)) if n else None))
    return result


def write_table(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(fields + [k for r in rows for k in r])))
        writer.writeheader()
        writer.writerows(rows)


def generate(run, events, static, selected, early=60, late=300):
    rows = compare_events(events, static, early, late)
    summary = summarize(rows, selected)
    write_table(run / 'punctuality_events.csv', rows,
                ['arrival_event_id', 'schedule_match_status', 'schedule_exclusion_reason',
                 'scheduled_arrival_timestamp', 'delay_seconds', 'delay_lower_seconds',
                 'delay_upper_seconds', 'punctuality_status', 'schedule_precision'])
    write_table(run / 'route_punctuality.csv', summary,
                ['route_id', 'route_name', 'direction_id', 'label_source', 'schedule_precision',
                 'matched_events', 'point_on_time_rate', 'on_time_rate_lower', 'on_time_rate_upper'])
    matched = [r for r in rows if r['schedule_match_status'] == 'matched']
    lines = ['', '## Schedule punctuality (exploratory)', '',
             f'On-time window: {-early/60:g} to +{late/60:g} minutes, inclusive. Research definition; not an asserted KAT policy.',
             f'Schedule-matched unique arrival events: {len(matched)}/{len(rows)}. No ETA is used as an actual arrival.',
             f'Schedule exclusions: {dict(Counter(r["schedule_exclusion_reason"] for r in rows if r["schedule_match_status"] != "matched"))}', '',
             '| Route / direction | Label source / schedule | Events (trips) | Point on-time rate | Arrival-uncertainty bounds | Uncertain events |',
             '|---|---|---:|---:|---:|---:|']
    for s in summary:
        point = f'{s["point_on_time_rate"]:.1%}' if s['matched_events'] else 'N/A'
        bounds = f'{s["on_time_rate_lower"]:.1%}–{s["on_time_rate_upper"]:.1%}' if s['matched_events'] else 'N/A'
        lines.append(f'| {s["route_name"]} / {s["direction_id"]} | {s["label_source"]} / {s["schedule_precision"]} | {s["matched_events"]} ({s["trip_instances"]}) | {point} | {bounds} | {s["uncertain_events"]} |')
    lines += ['', 'Rates count unique matched stop-arrival events, not prediction samples. Uncertain events stay in the denominator. Bounds reflect arrival-time uncertainty, not statistical confidence intervals.',
              'Estimated/interpolated stop times are reported separately from exact GTFS timepoints. Their schedule uncertainty is unknown and is not included in the arrival bounds.',
              'GPS labels estimate stop passage, not door opening. These selected vehicles and observed events are a small, non-random subset; missed arrivals and cancellations are not measured. Do not interpret this as system-wide punctuality.',
              'Positive delay means late; negative means early. The delay chart shows individual events with arrival-time bounds. The route overview pools schedule precision categories and shows conservative on-time counts; stratified statistics remain in the table and CSV.',
              'GTFS schedule reference: https://gtfs.org/documentation/schedule/reference/']
    draw_charts(run, matched, selected, static, early, late)
    return lines


def draw_charts(run, matched, selected, static, early, late):
    """Show individual delays and a plain-language route overview."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {'on_time': '#19846b', 'late': '#d15b32', 'early': '#347bb6', 'uncertain': '#7b8390'}
    names = {'on_time': 'On time', 'late': 'Late', 'early': 'Early', 'uncertain': 'Uncertain'}
    rule = f'Research on-time window: {early/60:g} min early to {late/60:g} min late. Not an official KAT standard.'
    estimated = sum(r['schedule_precision'] == 'estimated_stop_time' for r in matched)
    precision_note = f'Schedule: {estimated} estimated stop times; {len(matched)-estimated} exact timepoints.'
    source_note = 'GPS-inferred arrivals; small observed sample only.' if all(r['label_source'] == 'gps_inferred' for r in matched) else 'Observed/inferred arrivals; small observed sample only.'
    ordered = sorted(matched, key=lambda r: (r['route_id'], str(r['direction_id']), r['arrival_timestamp']))
    route_keys = sorted({(s['route_id'], str(s['direction_id'])) for s in selected} |
                        {(r['route_id'], str(r['direction_id'])) for r in matched})

    def route_label(route, direction):
        if static is not None and route in static.routes:
            number = static.routes[route].get('route_short_name') or route
        else:
            number = route
        return f'Route {number}'

    style = {'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
             'axes.unicode_minus': False, 'font.size': 11, 'text.color': '#253448',
             'axes.labelcolor': '#253448', 'xtick.color': '#637083', 'ytick.color': '#253448',
             'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False}
    with plt.rc_context(style):
        panel_count = max(1, len(route_keys))
        largest_panel = max((sum((r['route_id'], str(r['direction_id'])) == k for r in ordered) for k in route_keys), default=0)
        fig, axes = plt.subplots(1, panel_count, figsize=(max(12, 6.5*panel_count), max(7, 3.6+.32*largest_panel)), squeeze=False, sharex=True)
        fig.subplots_adjust(left=.08, right=.96, top=.72, bottom=.25, wspace=.32)
        fig.text(.055, .945, 'How late was each arrival?', fontsize=22, weight='bold')
        fig.text(.055, .895, rule, fontsize=11)
        fig.text(.055, .85, 'One dot = one inferred arrival. Separate panels show the unequal number of events per route.', fontsize=11, color='#637083')
        low = min([-early/60] + [r['delay_lower_seconds']/60 for r in ordered])
        high = max([late/60] + [r['delay_upper_seconds']/60 for r in ordered])
        span = max(5, high-low)
        for ax, key in zip(axes[0], route_keys or [None]):
            subset = [r for r in ordered if (r['route_id'], str(r['direction_id'])) == key]
            ax.axvspan(-early/60, late/60, color='#e5f3ec', zorder=0)
            ax.axvline(0, color='#8ba49a', linewidth=1)
            ax.axvline(late/60, color=colors['late'], linewidth=1, linestyle='--', alpha=.5)
            heading = f'{route_label(*key)} | {len(subset)} arrival'+('' if len(subset) == 1 else 's') if key else 'No selected routes'
            ax.set_title(heading, loc='left', fontsize=16, weight='bold', pad=20)
            labels = []
            for i, row in enumerate(subset):
                color = colors[row['punctuality_status']]
                value = row['delay_minutes']
                lo, hi = row['delay_lower_seconds']/60, row['delay_upper_seconds']/60
                marker = 's' if row['schedule_precision'] == 'exact_timepoint' else 'o'
                ax.plot([lo, hi], [i, i], color=color, linewidth=3, alpha=.3, solid_capstyle='round')
                ax.scatter([value], [i], color=color, s=75, marker=marker, zorder=3)
                value_text = f'{value:+.1f} min'
                ax.annotate(value_text, (hi, i), xytext=(9, 0), textcoords='offset points', va='center', fontsize=10, color=color)
                clock = datetime.fromtimestamp(row['arrival_timestamp'], static.zone).strftime('%H:%M:%S')
                labels.append(clock)
            ax.set_yticks(range(len(subset)), labels)
            ax.set_ylim(max(1, len(subset))-.5, -.5)
            ax.set_xlim(low-span*.06, high+span*.35)
            if not subset:
                ax.text(.5, .5, 'No matched arrival events', ha='center', va='center', transform=ax.transAxes)
            elif len(subset) == 1:
                ax.text(.04, .10, 'Only one event was identified.\nThis is not route-wide performance.', transform=ax.transAxes, fontsize=10, color='#637083')
            ax.tick_params(axis='y', length=0, pad=9, labelsize=9)
            ax.set_xlabel('Minutes vs. schedule (positive = late)', labelpad=12, fontsize=10)
            ax.grid(axis='x', color='#e7ebef', linewidth=.7)
            ax.set_axisbelow(True)
        fig.text(.055, .155, 'Y-axis: local arrival time. Green band: on-time window. Lines: arrival-time uncertainty.', fontsize=10, color='#637083')
        fig.text(.055, .105, 'Circles: estimated schedule times. Squares: exact timepoints. '+precision_note, fontsize=9, color='#637083')
        fig.text(.055, .06, source_note, fontsize=10, color='#637083')
        fig.savefig(run / 'delay_distribution.png', dpi=160, facecolor='white'); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, max(5.5, 3.8+len(route_keys)*.9)))
        fig.subplots_adjust(left=.18, right=.72, top=.69, bottom=.23)
        fig.text(.055, .94, 'How many observed arrivals were on time?', fontsize=21, weight='bold')
        fig.text(.055, .88, rule, fontsize=11)
        fig.text(.055, .825, f'{len(matched)} matched arrivals. Counts describe this sample, not long-term route performance.', fontsize=11, color='#637083')
        present = {r['punctuality_status'] for r in matched}
        legend = [Line2D([0], [0], color=colors[k], lw=9, label=names[k]) for k in colors if k in present]
        if legend:
            fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.055, .79), frameon=False, ncol=4, fontsize=10)
        overview = []
        for i, (route, direction) in enumerate(route_keys):
            subset = [r for r in matched if r['route_id'] == route and str(r['direction_id']) == direction]
            counts = Counter(r['punctuality_status'] for r in subset)
            n = len(subset)
            left = 0
            for status in colors:
                count = counts[status]
                if not count:
                    continue
                width = 100*count/n
                ax.barh(i, width, left=left, color=colors[status], height=.48)
                if width >= 15:
                    ax.text(left+width/2, i, f'{names[status]}: {count}', ha='center', va='center', color='white', fontsize=12, weight='bold')
                left += width
            text = f'{counts["on_time"]} / {n} on time' if n else 'No matched events'
            ax.text(1.04, i, text, transform=ax.get_yaxis_transform(), va='center', fontsize=15, weight='bold')
            if n:
                rate = 100*counts['on_time']/n
                details = ' · '.join(f'{names[k]} {counts[k]}' for k in colors if counts[k] and k != 'on_time')
                ax.annotate(f'{rate:.1f}% on time' + ('  |  '+details if details else ''),
                            (1.04, i), xycoords=ax.get_yaxis_transform(), xytext=(0, -22), textcoords='offset points', fontsize=10, color='#637083')
            overview.append(dict(route_id=route, direction_id=direction, matched_events=n,
                                 on_time_events=counts['on_time'], late_events=counts['late'],
                                 early_events=counts['early'], uncertain_events=counts['uncertain'],
                                 confirmed_on_time_rate=counts['on_time']/n if n else None))
        write_table(run / 'route_punctuality_overview.csv', overview, ['route_id', 'direction_id', 'matched_events', 'confirmed_on_time_rate'])
        ax.set_yticks(range(len(route_keys)), [route_label(*key)+f'\nDirection {key[1]}' for key in route_keys], fontsize=14)
        ax.tick_params(axis='y', length=0, pad=14)
        ax.set_ylim(max(1, len(route_keys))-.5, -.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        ax.set_xlabel('Share of observed arrivals', labelpad=12)
        if not route_keys:
            ax.set_ylim(-.5, .5)
            ax.text(.5, .5, 'No matched arrival events', ha='center', transform=ax.transAxes, fontsize=15)
        fig.text(.055, .105, 'On-time counts require the full arrival interval to be within the window. Uncertain events are counted separately.', fontsize=9, color='#637083')
        fig.text(.055, .065, source_note+' '+precision_note, fontsize=9, color='#637083')
        fig.text(.055, .025, 'Schedule types are pooled here; see route_punctuality.csv for separate groups.', fontsize=9, color='#637083')
        fig.savefig(run / 'route_punctuality.png', dpi=160, facecolor='white'); plt.close(fig)
