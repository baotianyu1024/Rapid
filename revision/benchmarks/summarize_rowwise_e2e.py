"""Collect completed GPU runs into a CSV without filling missing measurements."""
import csv
import json
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_rw_e2e_20261103')
OUT = Path('/home/longjiu.bty/Rapid/rowwise_e2e_results_20261103.csv')
SAMPLES = 510 * 65536
FIELDS = ('dataset', 'model', 'framework', 'local_batch', 'embedding_dim', 'steps',
          'fit_seconds', 'samples_per_second', 'auc', 'logloss', 'train_read_bytes_max',
          'peak_allocated_gib_max', 'process_seconds', 'rowwise_tables', 'log_path')


def metric(log, name):
    found = re.search(rf'^{re.escape(name)}=(.+)$', log, re.MULTILINE)
    return found.group(1) if found else None


def main():
    rows = []
    for dataset in ('terabyte', 'kaggle', 'taobao', 'movielens25m', 'movielens25m_user_temporal'):
        root = ROOT / dataset
        is_movie = dataset.startswith('movielens25m')
        models = ('dlrm', 'neumf') if is_movie else ('dcnv2', 'dlrm', 'wide_deep', 'deepfm')
        for model in models:
            unique_prefix = 'hugectr_unique' if dataset == 'taobao' or is_movie else 'hugectr_unique_verified'
            for framework, prefix in (('RAPID', 'rapid'), ('HugeCTR Unique', unique_prefix),
                                      ('HugeCTR Mixed20', 'hugectr_mixed20_verified')):
                path = root / f'{prefix}_{model}_b8192_d64_510.log'
                if not path.exists():
                    continue
                log = path.read_text(errors='replace')
                seconds = metric(log, 'FIT_SECONDS')
                result = metric(log, 'EVAL_METRICS')
                if not seconds or not result:
                    continue
                metrics = json.loads(result)
                rows.append(dict(dataset=dataset, model=model, framework=framework,
                                 local_batch=8192, embedding_dim=64, steps=510,
                                 fit_seconds=seconds,
                                 samples_per_second=f'{SAMPLES / float(seconds):.3f}',
                                 auc=metrics.get('AUC', ''), logloss='',
                                 train_read_bytes_max=metric(log, 'FIT_READ_BYTES') or '',
                                 peak_allocated_gib_max='',
                                 process_seconds=metric(log, 'PROCESS_WALL_SECONDS') or '',
                                 rowwise_tables='', log_path=str(path)))
            path = root / f'torchrec_rowwise_{model}_b8192_d64_510.json'
            plan = root / f'torchrec_rowwise_{model}_plan.json'
            if not path.exists() or not plan.exists():
                continue
            result = json.loads(path.read_text())
            entries = json.loads(plan.read_text())
            table_count = 2 if is_movie else 4 if dataset == 'taobao' else 26
            if len(entries) != table_count or any(
                    entry['sharding_type'] != 'row_wise' or entry['ranks'] != list(range(8))
                    for entry in entries.values()):
                raise ValueError(f'not all tables are row-wise: {plan}')
            eval_metrics = (result['eval_history'][-1] if is_movie
                            and result['eval_history'] else result['eval_last_batch'])
            if not eval_metrics:
                continue
            log_path = path.with_suffix('.log')
            log = log_path.read_text(errors='replace') if log_path.exists() else ''
            seconds = result['loop_wall_seconds']
            rows.append(dict(dataset=dataset, model=model, framework='TorchRec Row-wise',
                             local_batch=8192, embedding_dim=64, steps=510,
                             fit_seconds=f'{seconds:.6f}',
                             samples_per_second=f'{SAMPLES / seconds:.3f}',
                             auc=eval_metrics['auc'], logloss=eval_metrics['logloss'],
                             train_read_bytes_max=max(item['read_bytes'] for item in result['io_delta_by_rank']),
                             peak_allocated_gib_max=f'{max(result["peak_allocated_bytes_by_rank"]) / 2**30:.3f}',
                             process_seconds=metric(log, 'OUTER_PROCESS_SECONDS') or '',
                             rowwise_tables=len(entries), log_path=str(log_path)))
    with OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{OUT}: {len(rows)} completed runs')


if __name__ == '__main__':
    main()
