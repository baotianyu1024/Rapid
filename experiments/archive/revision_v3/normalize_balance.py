from pathlib import Path
import json,numpy as np
from report_comparisons import balance
V=Path(__file__).resolve().parents[1]
for name in ['criteo_8','criteo_64','kaggle_8','taobao_8','t180_8','t200_8']:
 d=V/'results'/name;p=d/'comparison.json';c=json.loads(p.read_text());cfg=c['config'];gm=json.loads((d/'gmp_traffic/het_gmp_s0.json').read_text());z=np.load(d/'gmp_traffic/het_gmp_s0.npz');f=z['forward'];back=z['backward'];full=np.arange(len(f))%gm['steps_per_epoch']<min(gm['rank_rows'])//cfg['local_batch'];c['methods']['HET-GMP'].update(balance(f.sum(2),back.sum(1),full,cfg['batches']*5));c['balance_scope']='All actual steps including ragged tails retained; mean and accumulated per-step maximum normalized by equal-work global-batch equivalents; absolute peak is an actual step.';p.write_text(json.dumps(c,indent=2)+'\n')
print('Normalized balance using every actual step, equal processed sample counts')
