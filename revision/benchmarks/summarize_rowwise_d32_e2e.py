"""Collect the uniform dim-32 GPU runs, rejecting non-row-wise TorchRec plans."""
import csv
import json
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_rw_d32_e2e_20261104')
OUT = Path('/home/longjiu.bty/Rapid/rowwise_d32_e2e_results_20261104.csv')
STEPS = 510
GLOBAL_BATCH = 65536
FIELDS = ('dataset', 'model', 'framework', 'optimizer', 'learning_rate', 'local_batch',
          'embedding_dim', 'steps', 'fit_seconds', 'samples_per_second', 'auc',
          'logloss', 'eval_rows', 'train_read_bytes_max', 'peak_allocated_gib_max',
          'process_seconds', 'rowwise_tables', 'log_path')


def metric(log, name):
    found = re.search(rf'^{re.escape(name)}=(.+)$', log, re.MULTILINE)
    return found.group(1) if found else None


def main():
    rows = []
    for dataset in ('terabyte', 'kaggle', 'taobao', 'movielens25m_user_temporal'):
        movie = dataset.startswith('movielens25m')
        models = ('dlrm', 'neumf') if movie else ('dcnv2', 'dlrm', 'wide_deep', 'deepfm')
        eval_rows = 5046272 if movie else 65536
        table_count = 2 if movie else 4 if dataset == 'taobao' else 26
        root = ROOT / dataset
        for model in models:
            methods = (('HugeCTR Unique', 'hugectr_unique'),) if movie else (
                ('HugeCTR Unique', 'hugectr_unique'), ('RAPID', 'rapid'))
            for framework, prefix in methods:
                path = root / f'{prefix}_{model}_b8192_d32_{STEPS}_adagrad.log'
                if not path.exists():
                    continue
                log = path.read_text(errors='replace')
                seconds = metric(log, 'FIT_SECONDS')
                evaluation = metric(log, 'EVAL_METRICS')
                if not seconds or not evaluation:
                    continue
                auc = json.loads(evaluation)['AUC']
                seconds = float(seconds)
                rows.append(dict(dataset=dataset, model=model, framework=framework,
                                 optimizer='AdaGrad', learning_rate=0.01, local_batch=8192,
                                 embedding_dim=32, steps=STEPS, fit_seconds=f'{seconds:.6f}',
                                 samples_per_second=f'{STEPS*GLOBAL_BATCH/seconds:.3f}',
                                 auc=f'{auc:.9f}', logloss='', eval_rows=eval_rows,
                                 train_read_bytes_max=metric(log, 'FIT_READ_BYTES') or '',
                                 peak_allocated_gib_max='',
                                 process_seconds=metric(log, 'PROCESS_WALL_SECONDS') or '',
                                 rowwise_tables='', log_path=str(path)))
            path = root / f'torchrec_rowwise_{model}_b8192_d32_{STEPS}_adagrad.json'
            plan = root / f'torchrec_rowwise_{model}_plan.json'
            if not path.exists() or not plan.exists():
                continue
            result = json.loads(path.read_text())
            entries = json.loads(plan.read_text())
            if len(entries) != table_count or any(
                    item['sharding_type'] != 'row_wise' or item['ranks'] != list(range(8))
                    for item in entries.values()):
                raise ValueError(f'non-row-wise TorchRec plan: {plan}')
            if result['embedding_dim'] != 32 or result['optimizer'] != 'adagrad':
                raise ValueError(f'configuration mismatch: {path}')
            evaluation = result['eval_history'][-1] if movie and result['eval_history'] else result['eval_last_batch']
            if not evaluation or evaluation['rows'] != eval_rows:
                raise ValueError(f'evaluation mismatch: {path}')
            seconds = result['loop_wall_seconds']
            log_path = path.with_suffix('.log')
            log = log_path.read_text(errors='replace') if log_path.exists() else ''
            rows.append(dict(dataset=dataset, model=model, framework='TorchRec Row-wise',
                             optimizer='AdaGrad', learning_rate=0.01, local_batch=8192,
                             embedding_dim=32, steps=STEPS, fit_seconds=f'{seconds:.6f}',
                             samples_per_second=f'{STEPS*GLOBAL_BATCH/seconds:.3f}',
                             auc=f'{evaluation["auc"]:.9f}', logloss=evaluation['logloss'],
                             eval_rows=eval_rows,
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
