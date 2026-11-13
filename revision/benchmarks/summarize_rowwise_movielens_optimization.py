"""Collect MovieLens optimizer checkpoints with their exact split and loss names."""
import csv
import json
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_rw_e2e_20261103')
OUT = Path('/home/longjiu.bty/Rapid/rowwise_movielens_optimization_20261103.csv')
SAMPLES_PER_STEP = 65536
FIELDS = ('split', 'model', 'framework', 'optimizer', 'learning_rate', 'steps',
          'fit_seconds', 'samples_per_second', 'auc', 'loss_metric', 'loss_value',
          'eval_rows', 'train_read_bytes_max', 'log_path')
CASES = (
    ('movielens25m', 'sgd', 0.001, 510, ''),
    ('movielens25m', 'sgd', 0.01, 5000, '_lr01'),
    ('movielens25m_user_temporal', 'sgd', 0.001, 510, ''),
    ('movielens25m_user_temporal', 'adagrad', 0.01, 510, '_adagrad'),
    ('movielens25m_user_temporal', 'adagrad', 0.01, 1000, '_adagrad'),
    ('movielens25m_user_temporal', 'adagrad', 0.01, 5000, '_adagrad'),
)


def metric(log, name):
    found = re.search(rf'^{name}=(.+)$', log, re.MULTILINE)
    return found.group(1) if found else None


def main():
    rows = []
    for split, optimizer, lr, steps, suffix in CASES:
        directory = ROOT / split
        eval_manifest = Path('/dev/shm') / (
            'rapid_movielens25m_user_temporal_20261103' if 'user_temporal' in split
            else 'rapid_movielens25m_temporal_20261103') / 'eval' / 'input.json'
        eval_rows = json.loads(eval_manifest.read_text())['rows']
        for model in ('dlrm', 'neumf'):
            for framework, prefix in (('HugeCTR Unique', 'hugectr_unique'),
                                      ('TorchRec Row-wise', 'torchrec_rowwise')):
                stem = f'{prefix}_{model}_b8192_d64_{steps}{suffix}'
                log_path = directory / f'{stem}.log'
                if not log_path.exists():
                    continue
                log = log_path.read_text(errors='replace')
                if framework == 'HugeCTR Unique':
                    seconds = metric(log, 'FIT_SECONDS')
                    raw = metric(log, 'EVAL_METRICS')
                    if not seconds or not raw:
                        continue
                    result = json.loads(raw)
                    loss_metric, loss = 'HugeCTR AverageLoss', result.get('AverageLoss', '')
                    auc = result['AUC']
                    read_bytes = metric(log, 'FIT_READ_BYTES') or ''
                else:
                    result_path = directory / f'{stem}.json'
                    plan = directory / f'torchrec_rowwise_{model}_{steps}{suffix}_plan.json'
                    if not plan.exists():
                        plan = directory / f'torchrec_rowwise_{model}_plan.json'
                    if not result_path.exists() or not plan.exists():
                        continue
                    entries = json.loads(plan.read_text())
                    if len(entries) != 2 or any(e['sharding_type'] != 'row_wise'
                                                or e['ranks'] != list(range(8))
                                                for e in entries.values()):
                        raise ValueError(f'invalid row-wise plan: {plan}')
                    result = json.loads(result_path.read_text())
                    evaluation = result['eval_history'][-1]
                    if evaluation['rows'] != eval_rows:
                        raise ValueError(f'evaluation rows mismatch: {result_path}')
                    seconds = result['loop_wall_seconds']
                    loss_metric, loss = 'LogLoss', evaluation['logloss']
                    auc = evaluation['auc']
                    read_bytes = max(item['read_bytes'] for item in result['io_delta_by_rank'])
                seconds = float(seconds)
                rows.append(dict(split=split, model=model, framework=framework,
                                 optimizer=optimizer, learning_rate=lr, steps=steps,
                                 fit_seconds=f'{seconds:.6f}',
                                 samples_per_second=f'{steps*SAMPLES_PER_STEP/seconds:.3f}',
                                 auc=f'{float(auc):.9f}', loss_metric=loss_metric,
                                 loss_value=loss, eval_rows=eval_rows,
                                 train_read_bytes_max=read_bytes, log_path=str(log_path)))
    with OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{OUT}: {len(rows)} runs')


if __name__ == '__main__':
    main()
