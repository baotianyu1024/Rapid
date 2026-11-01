"""Audit existing phase timers without relabeling mixed I/O stages as pure compute."""
import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1];old=root/'het_gmp_rapid/results/timing32_rerun';out=root/'results/speed_exact';out.mkdir(exist_ok=True)
stages=json.loads((old/'timing_summary.json').read_text());schedule=(old/'rapid_schedule.log').read_text();times={k:sum(map(float,re.findall(k+r'=([\d.]+)s',schedule))) for k in ['parse','hits','balance','write']};gmp=json.loads((old/'gmp/partition.json').read_text())
gmpcore=dict(graph=gmp['graph_seconds'],sample_refine=sum(x['data_seconds'] for x in gmp['rounds']),embedding_refine=sum(x['embedding_seconds'] for x in gmp['rounds']))
invalid=[]
for p in (root/'results').glob('*/materialization.json'):
 j=json.loads(p.read_text())
 if j.get('seconds',0)>100000:invalid.append(dict(path=str(p),invalid_seconds=j['seconds']))
report=dict(source_run=str(old),source_scope='historical sequential 32-rank 4,194,304-row fixed-configuration run; OMP16, RAPID scheduling async threads=1; not a new end-to-end rerun',
 rapid_schedule_phases=times,rapid_isolated_assignment_core_seconds=times['hits']+times['balance'],gmp_isolated_core_phases=gmpcore,gmp_isolated_core_seconds=sum(gmpcore.values()),
 gmp_priority_and_export_mixed_seconds=gmp['export_priority_seconds'],rapid_owner_and_map_archive_mixed_seconds=stages['rapid_owner_phase_times']['owner_compute_and_map_write_seconds'],
 original_total_wall_seconds=dict(rapid=stages['rapid_preprocess_seconds'],gmp=stages['gmp_preprocess_seconds']),
 original_final_record_materialize_seconds=stages['rapid_owner_phase_times']['materialize_seconds'],
 caveats=['Core timers exclude dataset writes, but do not cover all algorithm work: RAPID hot extraction/METIS/placement/owner and GMP priority sorting are additional.',
 'Do not call total minus known file writes an exact pure-compute total; several legacy stages mix computation, reading, compression, export and validation.',
 'User CPU seconds sum across threads, so they measure CPU work rather than elapsed algorithm time; GMP and RAPID have different effective parallelism.',
 'The new price optimizer is a postprocessing stage with a prior v7 warm start. Its kernel time is not full RAPID preprocessing time.',
 'The old materialize() total timer was overwritten by for-start loop indices; external stage timers and per-phase timers are unaffected. Fixed in current source.'],invalid_materialization_total_timers=invalid)
(out/'historical_time_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='invalid_materialization_total_timers'},indent=2))
