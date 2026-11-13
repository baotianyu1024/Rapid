"""Summarize completed paper-shape GPU runs without estimating missing cells."""

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path('/tmp/rapid_paper_table1_20261110')
PROJECT = Path('/home/longjiu.bty/Rapid')
SPECS = [('kaggle', 'dcn'), ('terabyte', 'dcnv2'), ('taobao', 'deepfm')]


def prior_csv(name, dataset, model):
    with (PROJECT / name).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['dataset'] == dataset and row['model'] == model:
                return float(row['samples_per_second'])
    return None


def huge(dataset, model, method):
    if method in ('reduction', 'unique', 'rapid'):
        log = Path('/tmp/rapid_reduction_vs_rapid_20261109') / f'{dataset}_{model}_{method}_510.log'
    else:
        log = ROOT / f'{dataset}_{model}_embark_510.log'
    match = re.search(r'^FIT_SECONDS=([\d.]+)$', log.read_text(), re.M)
    return 510 * 65536 / float(match[1]) if match else None


def torchrec(dataset, model):
    path = ROOT / f'{dataset}_{model}_torchrec_510.json'
    return json.loads(path.read_text())['samples_per_second'] if path.exists() else None


def gmp(dataset, model):
    if dataset == 'terabyte':
        return prior_csv('het_d32_holdout_results_20261104.csv', dataset, model)
    suffix = '_lr0001' if dataset == 'kaggle' else ''
    path = ROOT / f'{dataset}_{model}_gmp_510{suffix}.metrics'
    if not path.exists():
        return None
    blocks = re.findall(r'^TRAIN \d+: loss (\S+) .*?time ([\d.]+)', path.read_text(), re.M)
    if len(blocks) != 51 or any(not math.isfinite(float(loss)) for loss, _ in blocks):
        return None
    return 510 * 65536 / sum(float(seconds) for _, seconds in blocks)


def bagpipe(dataset, model):
    if dataset == 'terabyte':
        return prior_csv('bagpipe_d32_e2e_results_20261105.csv', dataset, model)
    directory = ROOT / 'bagpipe_full' / f'{dataset}_{model}_8gpu_510steps'
    if not directory.exists():
        return None
    durations = []
    for rank in range(8):
        logs = list(directory.glob(f'training_worker_{rank}_*.log'))
        if len(logs) != 1:
            return None
        content = logs[0].read_text()
        losses = [float(value) for value in re.findall(r'INFO:root:Loss ([^\n]+)', content)]
        match = re.search(r'Total Time of 510 iterations: ([\d.]+)ms', content)
        if len(losses) != 510 or not all(map(math.isfinite, losses)) or not match:
            return None
        durations.append(float(match[1]) / 1000)
    return 510 * 65536 / max(durations)


def herald(dataset, model):
    directory = ROOT / 'herald' / f'{dataset}_{model}_510steps'
    if not directory.exists():
        return prior_csv('herald_d32_e2e_results_20261105.csv', dataset, model) if dataset == 'terabyte' else None
    ranks = [directory / f'rank_{rank}.json' for rank in range(8)]
    if not all(path.exists() for path in ranks):
        return None
    values = [json.loads(path.read_text()) for path in ranks]
    if any(value['steps'] != 510 or not math.isfinite(value['last_loss']) for value in values):
        return None
    return 510 * 65536 / max(value['train_seconds'] for value in values)


for dataset, model in SPECS:
    baseline = 'unique' if dataset == 'taobao' else 'reduction'
    print(dataset, model, 'HugeCTR', huge(dataset, model, baseline),
          'TorchRec', torchrec(dataset, model), 'EMBark', huge(dataset, model, 'embark'),
          'HET-GMP', gmp(dataset, model), 'Herald', herald(dataset, model),
          'BagPipe', bagpipe(dataset, model), 'RAPID', huge(dataset, model, 'rapid'))
