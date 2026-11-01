from common import *
import ast
O=ROOT/'results/rank64_balance'
print('GMP/verification completed:',[p.stem.replace('.status','') for p in O.glob('*.status.json') if not p.name.startswith('rapid_')])
latest=O/'latest_rapid'
print('Latest RAPID completed:',[p.stem.replace('.status','') for p in latest.glob('*.status.json')])
for p in latest.glob('*.log'):
 if not p.with_suffix('.status.json').exists():print('Latest running',p.stem,p.read_text().splitlines()[-1][-150:] if p.stat().st_size else '')
f=O/'gmp_partition.log'
if f.exists():
 for line in reversed(f.read_text().splitlines()):
  if line.startswith("{'round'"):
   j=ast.literal_eval(line); print('GMP', {k:v for k,v in j.items() if k not in ['data_counts','embedding_counts']});
   for name in ['data_counts','embedding_counts']:
    a=j[name];print(name, 'min',min(a),'max',max(a),'max/mean',max(a)*len(a)/sum(a))
   break
  if line.startswith('graph constructed'):print(line);break
for name in ['rapid_hot','rapid_placement','rapid_schedule','rapid_owner_base','rapid_price_f2','rapid_price_b8','rapid_price_b32','rapid_guard_p5','rapid_final_materialize','gmp_replay','verify_outputs','report']:
 p=O/(name+'.log')
 if p.exists() and not (O/(name+'.status.json')).exists():print('Running',name, p.read_text().splitlines()[-1][-250:] if p.stat().st_size else '')
