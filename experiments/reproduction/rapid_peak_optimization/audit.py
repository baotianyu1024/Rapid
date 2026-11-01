import hashlib,json
from pathlib import Path
R=Path('/mnt/hugui/longjiu.bty/reproduction');root=R/'het_gmp_rapid/results/rank64_balance/latest_rapid'
m=json.load(open(root/'source_manifest.json'));hashes={k:dict(current=hashlib.sha256(Path(k).read_bytes()).hexdigest(),snapshot=hashlib.sha256(Path(v['snapshot']).read_bytes()).hexdigest(),recorded=v['sha256']) for k,v in m.items()}
assert all(len(set(x.values()))==1 for x in hashes.values())
stages=[]
for name in ['price_f2','price_b8','price_b32','guard_p5']:
 d=json.load(open(root/('rapid_'+name)/'refinement.json'))
 stages.append(dict(stage=name,parameters=d['parameters'],history=[dict(epoch=h['epoch'],moved=h['moved'],mean=h['forward']['mean'],recv_peak=h['forward']['mean_batch_peak'],send_peak=h['backward']['mean_batch_peak'],traffic_change=h['traffic_change']) for h in d['history']]))
result=dict(source_hashes_match=True,sources=hashes,stages=stages,findings=['All 12 current source hashes match the executed snapshot.', 'Peak refinement and guarded final stage were executed.', 'The first owner stage hits the per-batch 5% volume budget; later moves may be rejected even if they reduce a sender hotspot.', 'The guard preserves prior bounds rather than solving a global minimax objective.', 'Existing objective penalizes separate receive/send squared deviations, not the maximum combined per-rank load.'],latest_definition='current local source tree; not an online release')
(R/'rapid_peak_optimization/version_audit.json').write_text(json.dumps(result,indent=2))
print('version audit saved')
