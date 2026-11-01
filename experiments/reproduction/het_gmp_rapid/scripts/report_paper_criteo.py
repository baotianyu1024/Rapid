import hashlib,json
from pathlib import Path
from common import save
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reproduction/het_gmp_rapid/results/paper_criteo8'
read=lambda p:json.loads(p.read_text())
assert read(OUT/'complete.json')['success']
part=read(OUT/'native_partition.json');native={r['round']:r for r in part['rounds']};traces={r:read(OUT/f'trace_{r}.json') for r in [0,1,3,5]}
paper={0:277660,1:135942,3:103807,5:101425};rows=[]
for r in paper:
 n=native[r]['native_estimated_vectors_per_global_step'];t=traces[r]
 rows.append(dict(round=r,paper_value=paper[r],paper_reduction_percent=100*(1-paper[r]/paper[0]),native_value=n,native_reduction_percent=100*(1-n/native[0]['native_estimated_vectors_per_global_step']),exact_trace_full_step_mean=t['forward_full_step_mean'],exact_trace_epoch_vectors=t['forward_epoch_vectors'],exact_trace_epoch_reduction_percent=100*(1-t['forward_epoch_vectors']/traces[0]['forward_epoch_vectors']),float64_binomial_estimate=t['float64_binomial_expected_vectors_per_step'],native_vs_exact_percent=100*(n/t['forward_full_step_mean']-1)))
scope=read(OUT/'config.json')
save(OUT/'comparison.json',dict(config=scope,results=rows,paper_url='https://hsword.github.io/assets/pdf/sigmod2022-het-gmp.pdf',hardware='one visible Tesla V100S; no 8-GPU NCCL validation performed',metric='remote unique embedding vectors; 1D no mirrors, backward transpose has equal total; all full training rows included in epoch totals',limits=['One native run, not paper five-run mean; unspecified original shuffle seed and nondeterministic native OpenMP updates','Table3 text calls metric per epoch, while native code normalizes per global batch; retain both native values and explicit full-epoch trace counts','Company private dataset and Figure8 staleness100 not reproduced','Original raw Criteo, not RAPID modulo/multihot; no raw data dropped other than upstream 10% validation split']))
lines=['Criteo HET-GMP paper Table3 reproduction / estimator audit','',json.dumps(scope,ensure_ascii=False),'',
'8 logical ranks, B8192, unmodified native partition extension, snapshots random/1/3/5 rounds.',
'Actual ID trace enumeration is CPU counting, not network/GPU measurement. No mirrors. Backward total equals forward total.',
'Paper Table3 baseline numbers are compared separately from measured trace totals. Native get_communication is a float32 frequency estimator normalized to one global batch, despite the paper text referring to an epoch.',
'Round | Paper value | Paper reduction% | Native estimate | Native reduction% | Exact full-batch mean | Exact epoch vectors | Exact epoch reduction% | Float64 frequency estimate']
for r in rows:lines.append(' | '.join(str(round(r[k],4)) if isinstance(r[k],float) else str(r[k]) for k in ['round','paper_value','paper_reduction_percent','native_value','native_reduction_percent','exact_trace_full_step_mean','exact_trace_epoch_vectors','exact_trace_epoch_reduction_percent','float64_binomial_estimate']))
lines+=['','Caveats: raw vocabulary cardinality exactly matches paper; upstream seed unspecified, this run uses42; native seed1; one run, not five-run average. We cannot infer network bytes or NCCL runtime from these counts.','Paper87.5% Company/s100 claim not reproduced.','Native estimator vs trace gap may include float32 probability/accumulation error and batch ordering/sampling differences; float64 estimate shown as an independent diagnostic.']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
paths=[Path(__file__),ROOT/'reproduction/het_gmp_rapid/scripts/reproduce_paper_criteo.py',ROOT/'reproduction/het_gmp_rapid/scripts/paper_parse.cpp',ROOT/'reproduction/het_gmp_rapid/scripts/paper_parse',ROOT/'reproduction/het_gmp_rapid/scripts/hetuCTR_partition.cpython-310-x86_64-linux-gnu.so',ROOT/'reproduction/het_gmp_rapid/src/het-gmp/hetuCTR/csrc/partition/partition.cc']
save(OUT/'source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
