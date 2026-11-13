"""Summarize audited five-seed HugeCTR/RAPID short-train AUC pairs."""
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_r5_paired_d32_20261104')
OUT = Path('/home/longjiu.bty/Rapid')
FIELDS = ('dataset', 'model', 'method', 'seed', 'auc', 'logloss', 'fit_seconds', 'log_path')


def metric(text, name):
    match = re.search(rf'^{re.escape(name)}=(.+)$', text, re.M)
    return match.group(1) if match else None


def main():
    rows = []
    summary = []
    for dataset in ('kaggle', 'terabyte', 'taobao'):
        for model in ('dlrm', 'dcnv2', 'wide_deep', 'deepfm'):
            pairs = []
            for seed in (41, 42, 43, 44, 45):
                measured = {}
                for method in ('hugectr_unique', 'rapid'):
                    path = ROOT / f'{dataset}_{model}_{method}_seed{seed}_s510.log'
                    if not path.exists():
                        continue
                    log = path.read_text(errors='replace')
                    eval_json = metric(log, 'EVAL_METRICS')
                    exact_auc = metric(log, 'EVAL_AUC_EXACT')
                    loss = metric(log, 'EVAL_LOGLOSS')
                    seconds = metric(log, 'FIT_SECONDS')
                    if not (eval_json and exact_auc and loss and seconds):
                        continue
                    json.loads(eval_json)
                    auc = float(exact_auc)
                    if not (math.isfinite(auc) and 0 <= auc <= 1 and
                            math.isfinite(float(loss)) and float(loss) >= 0 and
                            math.isfinite(float(seconds)) and float(seconds) > 0):
                        continue
                    value = dict(dataset=dataset, model=model, method=method, seed=seed,
                                 auc=f'{auc:.9f}', logloss=f'{float(loss):.9f}',
                                 fit_seconds=f'{float(seconds):.6f}', log_path=str(path))
                    rows.append(value)
                    measured[method] = value
                if len(measured) == 2:
                    pairs.append(measured)
            if len(pairs) != 5:
                continue
            base = [float(pair['hugectr_unique']['auc']) for pair in pairs]
            rapid = [float(pair['rapid']['auc']) for pair in pairs]
            diff = [right - left for left, right in zip(base, rapid)]
            mean = sum(diff) / 5
            sd = math.sqrt(sum((value - mean)**2 for value in diff) / 4)
            half = 2.7764451051977987 * sd / math.sqrt(5)
            summary.append(dict(dataset=dataset, model=model, seeds=5,
                                reference_auc=f'{sum(base)/5:.9f}',
                                rapid_auc=f'{sum(rapid)/5:.9f}',
                                auc_diff=f'{mean:.9f}',
                                auc_diff_ci95_low=f'{mean-half:.9f}',
                                auc_diff_ci95_high=f'{mean+half:.9f}',
                                reference_logloss=f'{sum(float(pair["hugectr_unique"]["logloss"]) for pair in pairs)/5:.9f}',
                                rapid_logloss=f'{sum(float(pair["rapid"]["logloss"]) for pair in pairs)/5:.9f}'))
    for name, fields, records in (
        ('r5_paired_d32_runs_20261104.csv', FIELDS, rows),
        ('r5_paired_d32_summary_20261104.csv',
         ('dataset', 'model', 'seeds', 'reference_auc', 'rapid_auc', 'auc_diff',
          'auc_diff_ci95_low', 'auc_diff_ci95_high', 'reference_logloss',
          'rapid_logloss'), summary),
    ):
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f'{name}: {len(records)} rows')


if __name__ == '__main__':
    main()
