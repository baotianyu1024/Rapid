"""Summarize matched validation AUC and time-to-target from GPU logs."""
import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


TARGETS = (0.60, 0.65)
TIMESTAMP = re.compile(r'\[HCTR\]\[(\d\d:\d\d:\d\d\.\d+)\]')


def metric(text, name):
    match = re.search(rf'^{re.escape(name)}=([0-9.]+)$', text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_hugectr(path):
    text = path.read_text()
    fit_start = metric(text, 'FIT_START_PROCESS_SECONDS')
    if fit_start is None or 'PROCESS_WALL_SECONDS=' not in text:
        raise ValueError(f'incomplete HugeCTR run: {path}')
    points = []
    step = 0
    start_clock = None
    for line in text.splitlines():
        stamp = TIMESTAMP.search(line)
        if stamp and 'Use non-epoch mode' in line:
            start_clock = datetime.strptime(stamp.group(1), '%H:%M:%S.%f')
        iteration = re.search(r'Iter: (\d+) Time', line)
        if iteration:
            step = int(iteration.group(1))
        auc = re.search(r'Evaluation, AUC: ([0-9.]+)', line)
        if auc:
            if not stamp or not start_clock:
                raise ValueError(f'missing timestamp near AUC in {path}')
            when = datetime.strptime(stamp.group(1), '%H:%M:%S.%f')
            if when < start_clock:
                when += timedelta(days=1)
            points.append(dict(step=step, auc=float(auc.group(1)),
                               process_wall_seconds=fit_start + (when - start_clock).total_seconds()))
        loss = re.search(r'Evaluation, AverageLoss: ([0-9.]+)', line)
        if loss and points:
            points[-1]['hugectr_average_loss'] = float(loss.group(1))
    return dict(points=points, fit_seconds=metric(text, 'FIT_SECONDS'),
                fit_read_bytes=int(metric(text, 'FIT_READ_BYTES')),
                process_wall_seconds=metric(text, 'PROCESS_WALL_SECONDS'),
                input_setup_seconds=metric(text, 'INPUT_SETUP_SECONDS'),
                compile_seconds=metric(text, 'MODEL_COMPILE_SECONDS'))


def parse_torchrec(path):
    data = json.loads(path.read_text())
    return dict(points=data['eval_history'], process_wall_seconds=data['eval_history'][-1]['process_wall_seconds'],
                mean_step_seconds=data['mean_step_seconds'],
                samples_per_second=data['samples_per_second'],
                max_gpu_allocated_gib=max(data['peak_allocated_bytes_by_rank']) / 2**30,
                read_bytes_by_rank=[x['read_bytes'] for x in data['io_delta_by_rank']])


def first_targets(points):
    return {str(target): next((point for point in points if point['auc'] >= target), None)
            for target in TARGETS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--torchrec-steps', type=int)
    args = parser.parse_args()
    torchrec_steps = args.torchrec_steps or args.steps
    results = {}
    for name, filename, parser_fn in [
        ('HugeCTR', f'hugectr_native_ram_{args.steps}.log', parse_hugectr),
        ('TorchRec', f'torchrec_ram_{torchrec_steps}.json', parse_torchrec),
        ('RAPID+HugeCTR', f'hugectr_rapid_ram_{args.steps}.log', parse_hugectr),
    ]:
        path = args.root / filename
        if not path.exists():
            continue
        result = parser_fn(path)
        result['targets'] = first_targets(result['points'])
        results[name] = result
    args.out.write_text(json.dumps(results, indent=2) + '\n')
    for name, result in results.items():
        print(name, 'final_auc', result['points'][-1]['auc'], 'time_to_targets', result['targets'])


if __name__ == '__main__':
    main()
