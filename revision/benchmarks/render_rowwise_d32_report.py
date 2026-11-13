"""Render the dim-32 comparison directly from measured run records."""
import csv
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
CSV = ROOT / 'rowwise_d32_e2e_results_20261104.csv'
REPORT = ROOT / 'RowWise_Dim32_端到端复测_20261104.md'
START = '<!-- RESULTS_START -->'
END = '<!-- RESULTS_END -->'
DATASETS = {
    'terabyte': ('Terabyte', ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')),
    'kaggle': ('Kaggle', ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')),
    'taobao': ('Taobao', ('dlrm', 'dcnv2', 'wide_deep', 'deepfm')),
    'movielens25m_user_temporal': ('MovieLens-25M', ('dlrm', 'neumf')),
}
MODELS = {'dlrm': 'DLRM', 'dcnv2': 'DLRM-DCNv2', 'wide_deep': 'Wide&Deep',
          'deepfm': 'DeepFM', 'neumf': 'NeuMF'}
METHODS = ('RAPID', 'HugeCTR Unique', 'TorchRec Row-wise')


def main():
    measurements = {}
    for row in csv.DictReader(CSV.open()):
        if (int(row['embedding_dim']), int(row['local_batch']), int(row['steps']),
                row['optimizer']) != (32, 8192, 510, 'AdaGrad'):
            raise ValueError(f'inconsistent configuration: {row}')
        key = row['dataset'], row['model'], row['framework']
        if key in measurements:
            raise ValueError(f'duplicate run: {key}')
        measurements[key] = row
    throughput = [
        '| Dataset / Model | RAPID samples/s | RAPID AUC | HugeCTR samples/s | HugeCTR AUC | TorchRec row-wise samples/s | TorchRec AUC |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    processes = [
        '| Dataset / Model | RAPID process s | HugeCTR process s | TorchRec process s |',
        '| --- | ---: | ---: | ---: |',
    ]
    complete = 0
    expected = 0
    for dataset, (label, models) in DATASETS.items():
        for model in models:
            cells = [f'{label} / {MODELS[model]}']
            times = [cells[0]]
            for method in METHODS:
                row = measurements.get((dataset, model, method))
                if dataset.startswith('movielens25m') and method == 'RAPID':
                    cells.extend(('N/A (no CPS/ACP plan)', 'N/A'))
                    times.append('N/A')
                    continue
                expected += 1
                if row is None:
                    cells.extend(('pending', 'pending'))
                    times.append('pending')
                    continue
                complete += 1
                cells.extend((f'{float(row["samples_per_second"]):,.0f}',
                              f'{float(row["auc"]):.6f}'))
                times.append(f'{float(row["process_seconds"]):.3f}' if row['process_seconds'] else 'unavailable')
            throughput.append('| ' + ' | '.join(cells) + ' |')
            processes.append('| ' + ' | '.join(times) + ' |')
    body = '\n'.join(throughput + ['', '### Process time (startup + training + evaluation)', ''] + processes)
    source = REPORT.read_text()
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('report markers missing or duplicated')
    prefix, remainder = source.split(START, 1)
    _, suffix = remainder.split(END, 1)
    REPORT.write_text(prefix + START + '\n' + body + '\n' + END + suffix)
    print(f'{complete}/{expected} measured configurations rendered to {REPORT}')


if __name__ == '__main__':
    main()
