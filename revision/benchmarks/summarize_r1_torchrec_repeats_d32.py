"""Compare new TorchRec seeds with the original seed-42 R1 measurements."""
import csv
import json
import statistics
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
NEW = Path('/tmp/rapid_r1_torchrec_repeats_d32_20261104')
OUT = ROOT / 'torchrec_rowwise_d32_repeats_20261104.csv'


def main():
    original = {(row['dataset'], row['model']): float(row['samples_per_second'])
                for row in csv.DictReader((ROOT / 'rowwise_d32_e2e_results_20261104.csv').open())
                if row['framework'] == 'TorchRec Row-wise' and row['dataset'] in
                ('kaggle', 'terabyte', 'taobao')}
    rows = []
    for dataset in ('kaggle', 'terabyte', 'taobao'):
        for model in ('dlrm', 'dcnv2', 'wide_deep', 'deepfm'):
            values = {42: original[(dataset, model)]}
            for seed in (43, 44):
                path = NEW / f'{dataset}_{model}_seed{seed}.json'
                if path.exists():
                    result = json.loads(path.read_text())
                    if result.get('eval_last_batch'):
                        values[seed] = float(result['loop_samples_per_second'])
            if len(values) == 3:
                rows.append(dict(dataset=dataset, model=model, seed42=f'{values[42]:.3f}',
                                 seed43=f'{values[43]:.3f}', seed44=f'{values[44]:.3f}',
                                 mean=f'{statistics.mean(values.values()):.3f}',
                                 sd=f'{statistics.stdev(values.values()):.3f}'))
    with OUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('dataset', 'model', 'seed42',
                                                     'seed43', 'seed44', 'mean', 'sd'))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{OUT}: {len(rows)} complete three-run groups')


if __name__ == '__main__':
    main()
