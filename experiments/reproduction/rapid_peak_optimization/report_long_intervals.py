"""Report equal-horizon traffic savings, including modeled migration costs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'long_interval_report'
control = json.loads((ROOT / 'stream64_phase_control/metrics.json').read_text())
summaries, rounds = [], []
for interval in [0, 4, 8, 12, 16]:
    folder = ROOT / ('stream64_phase_control' if interval == 0 else f'stream64_phase_i{interval}_f001')
    assert json.loads((folder / 'complete.json').read_text())['success']
    data = json.loads((folder / 'metrics.json').read_text())
    assert len(data) == len(control) == 20
    gross = migration = 0
    costs = []
    for row, baseline in zip(data, control):
        for key in ['window', 'source_row_start', 'source_row_end', 'rows', 'steps']:
            assert row[key] == baseline[key]
        saved = (baseline['training_weight_bytes'] + baseline['training_metadata_bytes']
                 - row['training_weight_bytes'] - row['training_metadata_bytes'])
        cost = row['migration_adam_plus_routes_bytes']
        gross += saved
        migration += cost
        if row['migration_moved_rows']:
            costs.append({'before_window': row['window'], 'moved_rows': row['migration_moved_rows'],
                          'bytes': cost})
        rounds.append(dict(interval_windows=interval, window=row['window'],
            peak=row['peak_combined_receive'], baseline_peak=baseline['peak_combined_receive'],
            total_vectors_per_rank_step=row['total_vectors_per_8192'],
            cv_percent=row['rank_communication_cv'] * 100,
            gross_savings_GiB=saved / 2**30, migration_GiB=cost / 2**30,
            cumulative_net_GiB=(gross - migration) / 2**30))
    # Fixed common scoring horizon, including the period before later policies migrate.
    peak = sum(x['peak_combined_receive'] for x in data[4:]) / 16
    base_peak = sum(x['peak_combined_receive'] for x in control[4:]) / 16
    total = sum(x['total_vectors_per_8192'] for x in data[4:]) / 16
    base_total = sum(x['total_vectors_per_8192'] for x in control[4:]) / 16
    after = data[interval:] if interval else []
    base_after = control[interval:] if interval else []
    summaries.append(dict(interval_windows=interval, interval_training_steps=interval * 20,
        migration_events=len(costs), moved_rows=sum(x['moved_rows'] for x in costs),
        gross_training_savings_GiB=gross / 2**30, migration_Adam_routes_GiB=migration / 2**30,
        net_savings_GiB=(gross - migration) / 2**30,
        net_savings_percent_all20=100 * (gross - migration) / sum(
            x['training_weight_bytes'] + x['training_metadata_bytes'] for x in control),
        mean_peak_windows5_20=peak, peak_reduction_percent=(1 - peak / base_peak) * 100,
        total_reduction_percent=(1 - total / base_total) * 100,
        post_first_migration_windows=len(after),
        post_first_migration_peak_reduction_percent=(
            (1 - sum(x['peak_combined_receive'] for x in after)
             / sum(x['peak_combined_receive'] for x in base_after)) * 100 if after else None),
        checkpoints=costs))
for name, values in [('policy_comparison', summaries), ('window_comparison', rounds)]:
    (OUT / f'{name}.json').write_text(json.dumps(values, indent=2))
    flat = [{k: v for k, v in x.items() if k != 'checkpoints'} for x in values]
    with (OUT / f'{name}.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
(OUT / 'methodology.txt').write_text(
    'Same day6 D0 and 20 ordered incremental windows, 64 ranks, 2097152 rows/window, '
    '8192 samples/rank/global batch, five epochs/window. Only migration interval changes; '
    'each event budget is 0.001 of known embeddings, counting both swap endpoints. '
    'Decisions use the immediately preceding window, not all windows since the last event. '
    'No sample crosses an original global batch. Prefixes are checked against the control '
    'for identical assignments and all four traffic matrices. New final-window raw replays '
    'and final ownership/physical-slot audits run independently. '
    'Common peak/total score uses windows5..20; net savings uses all20 windows. '
    'Training savings includes tensor and metadata traffic. Migration includes128FP32 weights, '
    'two Adam states and16-byte route records replicated to63 peers. '
    'CPU communication replay only; migration bytes modeled, not GPU transport or elapsed-time savings. '
    'Intervals12 and16 produce only one event each; this does not establish repeated long-term behavior.\n')
print(json.dumps(summaries, indent=2))
