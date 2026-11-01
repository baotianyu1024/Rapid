"""Rebuild saved case summaries from immutable count tensors under receive-only cost."""
import json
import shutil
from pathlib import Path

import numpy as np

from measure import summarize_counts

ROOT=Path(__file__).resolve().parents[1]


def source_counts(case, tag, batches):
    current=case
    while True:
        p=current/f'{tag}_counts.npz'
        if p.exists():
            arrays=np.load(p)
            if arrays['receive'].shape[0]>=batches:
                return {k:arrays[k][:batches] for k in ['receive','send','unique','hits']}
        cfg=json.loads((current/'config.json').read_text())
        if not cfg.get('base_case'):
            raise FileNotFoundError(f'No {tag} counts covering {case} ({batches} batches)')
        current=Path(cfg['base_case'])


def main():
    cases=sorted(p for p in (ROOT/'results').iterdir() if (p/'complete.json').exists())
    for case in cases:
        cfg=json.loads((case/'config.json').read_text())
        for tag in ('hash','cps'):
            counts=source_counts(case,tag,cfg['batches'])
            old=case/f'{tag}_metrics.json'
            backup=case/f'{tag}_metrics_legacy_bg.json'
            if not backup.exists():shutil.copyfile(old,backup)
            old_data=json.loads(old.read_text())
            result=summarize_counts(**{'recv':counts['receive'],'send':counts['send'],
                'unique':counts['unique'],'hits':counts['hits'],
                'ranks':cfg['ranks'],'batch':cfg['batch_size'],
                'lengths':cfg['table_lengths'],'cps':tag=='cps',
                'paths':old_data.get('paths',[])})
            old.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Recomputed receive-only metrics for {len(cases)} cases; legacy summaries retained')


if __name__=='__main__':main()
