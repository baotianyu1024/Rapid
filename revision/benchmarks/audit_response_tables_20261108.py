"""Read-only independent recomputation of response tables and raw-log evidence."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'response_audit_20261108'


def records(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    text = (ROOT / '中文Response.md').read_text()
    issues, checks = [], []
    table1, table5 = {}, {}
    for line in text.splitlines():
        if not line.startswith('| '):
            continue
        cells = [x.strip() for x in line.split('|')[1:-1]]
        if '／' in cells[0]:
            if len(cells) == 8:
                table1[cells[0]] = cells
            if len(cells) == 5:
                table5[cells[0]] = cells
    datasets = {'kaggle': 'Kaggle', 'terabyte': 'Terabyte', 'taobao': 'Taobao'}
    models = {'dlrm': 'DLRM', 'dcnv2': 'DLRM-DCNv2', 'wide_deep': 'Wide&Deep', 'deepfm': 'DeepFM'}
    sources = [
        ('rowwise_d32_e2e_results_20261104.csv', None),
        ('embark_d32_e2e_results_20261104.csv', 3),
        ('het_d32_holdout_results_20261104.csv', 4),
        ('het_d32_multimodel_results_20261104.csv', 4),
        ('het_kaggle_deepfm_stable_bce_20261105.csv', 4),
        ('herald_d32_e2e_results_20261105.csv', 5),
        ('bagpipe_d32_e2e_results_20261105.csv', 6),
    ]
    r1 = {}
    for source, column in sources:
        for row in records(source):
            if row['dataset'] not in datasets:
                continue
            label = datasets[row['dataset']] + '／' + models[row['model']]
            col = column if column is not None else {
                'HugeCTR Unique': 1, 'TorchRec Row-wise': 2, 'RAPID': 7,
            }.get(row.get('framework'))
            if col is None or not row.get('samples_per_second'):
                continue
            r1[label, col] = (row, source)
    for (label, col), (row, source) in r1.items():
        actual = float(row['samples_per_second'])
        displayed = float(re.match(r'[\d,]+', table1[label][col])[0].replace(',', ''))
        if abs(actual - displayed) > .501:
            issues.append(dict(table='R1', label=label, column=col, csv=actual, response=displayed))
        checks.append(dict(table='R1', label=label, column=col, csv=source, value=actual))
    runs = records('r5_paired_d32_runs_20261104.csv')
    summary = {(r['dataset'], r['model']): r for r in records('r5_paired_d32_summary_20261104.csv')}
    pairs = {}
    evidence = []
    for row in runs:
        key = row['dataset'], row['model']
        pairs.setdefault(key, {}).setdefault(int(row['seed']), {})[row['method']] = row
        path = Path(row['log_path'])
        if not path.exists():
            evidence.append(dict(path=str(path), status='missing'))
            issues.append(dict(table='R5 raw log', path=str(path), status='missing'))
            continue
        log = path.read_text(errors='replace')
        for field, marker in [('auc', 'EVAL_AUC_EXACT'), ('logloss', 'EVAL_LOGLOSS'), ('fit_seconds', 'FIT_SECONDS')]:
            match = re.search(rf'^{marker}=(.+)$', log, re.M)
            if not match or abs(float(match[1]) - float(row[field])) > 1e-8:
                issues.append(dict(table='R5 raw log', path=str(path), field=field))
        evidence.append(dict(path=str(path), status='checked'))
    recomputed = []
    for key, seeds in pairs.items():
        assert set(seeds) == {41, 42, 43, 44, 45}
        base = [float(seeds[s]['hugectr_unique']['auc']) for s in sorted(seeds)]
        rapid = [float(seeds[s]['rapid']['auc']) for s in sorted(seeds)]
        delta = [b-a for a, b in zip(base, rapid)]
        half = 2.7764451051977987 * stdev(delta) / math.sqrt(5)
        values = dict(reference_auc=mean(base), rapid_auc=mean(rapid), auc_diff=mean(delta),
                      auc_diff_ci95_low=mean(delta)-half, auc_diff_ci95_high=mean(delta)+half,
                      reference_logloss=mean(float(v['hugectr_unique']['logloss']) for v in seeds.values()),
                      rapid_logloss=mean(float(v['rapid']['logloss']) for v in seeds.values()))
        for field, value in values.items():
            if abs(value - float(summary[key][field])) > 2e-9:
                issues.append(dict(table='R5 summary', key=key, field=field, recomputed=value))
        label = datasets[key[0]] + '／' + models[key[1]]
        displayed = [float(s) for cell in table5[label][1:] for s in re.findall(r'[+-]?\d+\.\d+', cell)]
        expected = [values[f] for f in ('reference_auc', 'rapid_auc', 'auc_diff',
                    'auc_diff_ci95_low', 'auc_diff_ci95_high', 'reference_logloss', 'rapid_logloss')]
        if len(displayed) != len(expected):
            issues.append(dict(table='R5 response', label=label, status='unexpected field count'))
        for i, (a, b) in enumerate(zip(displayed, expected)):
            if abs(a - b) > .000000501:
                issues.append(dict(table='R5 response', label=label, field=i, response=a, recomputed=b))
        recomputed.append(dict(dataset=key[0], model=key[1], **values))
    if len(r1) != 84 or len(recomputed) != 12:
        issues.append(dict(table='coverage', r1_cells=len(r1), r5_groups=len(recomputed)))
    result = dict(r1_cells_checked=len(r1),
                  r5_raw_runs_checked=sum(r['status'] == 'checked' for r in evidence),
                  r5_pairs_recomputed=len(recomputed), issues=issues, r1=checks,
                  r5=recomputed, evidence=evidence)
    (out / 'table_recomputation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k in
                     ('r1_cells_checked', 'r5_raw_runs_checked', 'r5_pairs_recomputed', 'issues')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
