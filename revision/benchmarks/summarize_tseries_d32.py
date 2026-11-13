"""Collect verified dim-32 T-series communication and GPU training measurements."""
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path('/tmp/rapid_tseries_d32_20261104')
GPU = Path('/tmp/rapid_tseries_d32_gpu_20261104')
OUT = Path('/home/longjiu.bty/Rapid')
sys.path.insert(0, '/home/longjiu.bty/Rapid/code/Rapid/experiments/baselines')
from torchrec_count import table_bytes


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def metric(log, name):
    found = re.search(rf'^{re.escape(name)}=(.+)$', log, re.MULTILINE)
    return float(found.group(1)) if found else None


def main():
    communication = []
    training = []
    for dataset, case_name, tables in (('T180', 't180', 177), ('T200', 't200', 200),
                                       ('T510', 't510_verified', 510)):
        case = ROOT / case_name
        meta = read(case / 'data/input.json')
        official = read(Path(f'/tmp/rapid_official_{dataset.lower()}_20261103/manifest.json'))
        if meta is None or official is None or meta['input_sha256'] != official['sha256']:
            raise ValueError(f'input SHA mismatch: {dataset}')
        if (meta['rows'], meta['ranks'], meta['local_batch'], len(meta['lengths'])) != (65536, 8, 512, tables):
            raise ValueError(f'input configuration mismatch: {dataset}')
        sources = (
            ('RAPID', case / 'communication.json',
             lambda x: x['all_rank_forward_backward_GiB'] / 4),
            ('HET-GMP', case / 'gmp_traffic/het_gmp_s0.json',
             lambda x: x['vector_bytes_32'] / 2**30),
            ('Herald', case / 'herald/result.json',
             lambda x: x['vector_bytes_32'] / 2**30),
            ('BagPipe', case / 'bagpipe/result.json',
             lambda x: x['total_embedding_bytes'] / 2**30),
        )
        for method, path, extract in sources:
            value = read(path)
            if value is None:
                continue
            if method == 'RAPID' and (value['dim'], value['epochs'], value['gpu_execution']) != (128, 5, False):
                raise ValueError(f'RAPID replay mismatch: {path}')
            if method == 'BagPipe' and (value['dim'], value['epochs'], value['steps']) != (32, 5, 80):
                raise ValueError(f'BagPipe replay mismatch: {path}')
            communication.append(dict(dataset=dataset, method=method, dimension=32,
                                      ranks=8, local_batch=512, epochs=5,
                                      embedding_payload_gib=f'{extract(value):.9f}',
                                      input_sha256=meta['input_sha256'], evidence=str(path)))
        plan_path = case / 'torchrec/plan_d32_rowwise.json'
        plan = read(plan_path)
        if plan is not None:
            entries = plan['tables']
            if (len(entries) != tables or plan['dim'] != 32 or plan['optimizer'] != 'adagrad'
                    or any(item['sharding_type'] != 'row_wise' or
                           sorted(item['ranks']) != list(range(8)) for item in entries)):
                raise ValueError(f'TorchRec is not all row-wise: {plan_path}')
            payload = sum(item['forward_bytes'] + item['backward_bytes'] for item in
                          (table_bytes(table, 8, 512, 80) for table in entries))
            communication.append(dict(dataset=dataset, method='TorchRec Row-wise',
                                      dimension=32, ranks=8, local_batch=512, epochs=5,
                                      embedding_payload_gib=f'{payload / 2**30:.9f}',
                                      input_sha256=meta['input_sha256'], evidence=str(plan_path)))
        torch_path = GPU / f'{dataset.lower()}_torchrec_510.json'
        torch = read(torch_path)
        if torch is not None:
            seconds = torch['loop_wall_seconds']
            training.append(dict(dataset=dataset, method='TorchRec Row-wise',
                                 optimizer='AdaGrad', dimension=32, local_batch=512, steps=510,
                                 fit_seconds=f'{seconds:.6f}',
                                 samples_per_second=f'{510 * 4096 / seconds:.3f}',
                                 train_read_bytes_max=max(item['read_bytes'] for item in torch['io_delta_by_rank']),
                                 evidence=str(torch_path)))
        for method, stem in (('HugeCTR Unique', f'{dataset.lower()}_hugectr_unique_adagrad_d32_b512_510'),
                             ('HugeCTR Hash ACP', f'{dataset.lower()}_hugectr_hash_acp_adagrad_d32_b512_510'),
                             ('RAPID', f'{dataset.lower()}_rapid_adagrad_d32_b512_510')):
            path = GPU / f'{stem}.log'
            if method == 'RAPID' and dataset == 'T180':
                path = GPU / f'{stem}_shm.log'
            if not path.exists():
                continue
            log = path.read_text(errors='replace')
            seconds = metric(log, 'FIT_SECONDS')
            if seconds is None:
                continue
            training.append(dict(dataset=dataset, method=method, optimizer='AdaGrad',
                                 dimension=32, local_batch=512, steps=510,
                                 fit_seconds=f'{seconds:.6f}',
                                 samples_per_second=f'{510 * 4096 / seconds:.3f}',
                                 train_read_bytes_max=metric(log, 'FIT_READ_BYTES'), evidence=str(path)))
    for name, fields, records in (
        ('tseries_d32_communication_20261104.csv', ('dataset', 'method', 'dimension', 'ranks',
          'local_batch', 'epochs', 'embedding_payload_gib', 'input_sha256', 'evidence'), communication),
        ('tseries_d32_gpu_training_20261104.csv', ('dataset', 'method', 'optimizer', 'dimension',
          'local_batch', 'steps', 'fit_seconds', 'samples_per_second', 'train_read_bytes_max',
          'evidence'), training),
    ):
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f'{name}: {len(records)} records')


if __name__ == '__main__':
    main()
