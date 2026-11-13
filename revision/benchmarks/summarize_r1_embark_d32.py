"""Summarize actual GPU runs using the upstream EMBark auto planner."""
import csv
import json
import re
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
LOGS = Path('/tmp/rapid_r1_embark_d32_20261104')
OUT = ROOT / 'embark_d32_e2e_results_20261104.csv'
FIELDS = ('dataset', 'model', 'seed', 'steps', 'fit_seconds',
          'samples_per_second', 'fit_read_bytes', 'unique_tables',
          'reduction_tables', 'log_path')


def value(text, name):
    match = re.search(rf'^{re.escape(name)}=(.+)$', text, re.M)
    return match.group(1) if match else None


def main():
    rows = []
    for path in sorted(LOGS.glob('*.log')):
        parts = path.stem.split('_')
        if len(parts) < 4:
            continue
        dataset, seed, steps = parts[0], int(parts[-2][4:]), int(parts[-1][1:])
        if steps != 510:
            continue
        model = '_'.join(parts[1:-2])
        log = path.read_text(errors='replace')
        if re.search(r'Loss:\s*(?:nan|inf)', log, re.I):
            continue
        seconds = value(log, 'FIT_SECONDS')
        read_bytes = value(log, 'FIT_READ_BYTES')
        plan = value(log, 'EMBARK_PLAN')
        if not (seconds and read_bytes and plan):
            continue
        plan = json.loads(plan)
        rows.append(dict(dataset=dataset, model=model, seed=seed, steps=steps,
                         fit_seconds=seconds,
                         samples_per_second=f'{steps * 65536 / float(seconds):.3f}',
                         fit_read_bytes=read_bytes,
                         unique_tables=len(plan['unique']),
                         reduction_tables=len(plan['reduction']), log_path=str(path)))
    with OUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{OUT}: {len(rows)} runs')


if __name__ == '__main__':
    main()
