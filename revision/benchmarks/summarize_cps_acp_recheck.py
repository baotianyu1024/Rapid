"""Summarize fixed-policy HugeCTR runs without conflating GPU and wall time."""
import datetime as dt
import json
import re
import statistics
from pathlib import Path

root = Path('/tmp/rapid_acp_recheck')
results = {}
for prefix in ('', 'online_'):
    for mode in ('remote_acp', 'forced_table20_reduction'):
        key = prefix + mode
        runs = []
        for run in (1, 2, 3):
            text = (root / f'{key}_{run}.log').read_text()
            moments = {int(n): dt.datetime.strptime(t, '%H:%M:%S.%f') for t, n in
                       re.findall(r'\[HCTR\]\[(\d\d:\d\d:\d\d\.\d+)\].*Iter: (\d+)', text)}
            elapsed = (moments[110] - moments[30]).total_seconds()
            io = [int(re.search(rf'IO_PROFILE_{tag} read_bytes: (\d+)', text).group(1)) for tag in ('START', 'END')]
            setup = re.search(r'INPUT_SETUP_SECONDS=([0-9.]+)', text)
            runs.append(dict(step_seconds=elapsed / 80, samples_per_second=65536 * 80 / elapsed,
                             physical_read_bytes=io[1] - io[0],
                             input_setup_seconds=float(setup.group(1)) if setup else None,
                             losses={int(n): float(v) for n, v in re.findall(r'Iter: (\d+) .*?Loss: ([0-9.eE+-]+)', text)}))
        results[key] = dict(runs=runs, mean_samples_per_second=statistics.mean(x['samples_per_second'] for x in runs))
    left, right = results[prefix + 'remote_acp']['runs'], results[prefix + 'forced_table20_reduction']['runs']
    for a, b in zip(left, right):
        if a['physical_read_bytes'] or b['physical_read_bytes']:
            raise ValueError('physical read during measurement')
        if a['losses'].keys() != b['losses'].keys():
            raise ValueError('mismatched printed loss points')
        a['max_printed_loss_difference'] = max(abs(a['losses'][n] - b['losses'][n]) for n in a['losses'])
(root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
for name, value in results.items():
    print(name, [round(x['step_seconds'] * 1000, 3) for x in value['runs']],
          round(value['mean_samples_per_second']))
