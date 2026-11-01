from pathlib import Path
import os,sys,json,time,signal,subprocess
V=Path(__file__).resolve().parents[1];O=V/'results/long_incremental';E=V/'results/incremental_extension'
if len(sys.argv)>1 and sys.argv[1]=='worker':
 k=int(sys.argv[2]);src=(V/'code/long_incremental.py').read_text().split('initial=OwnerState(g,26);')[0].replace('out.mkdir(exist_ok=False)','out.mkdir(exist_ok=True)');exec(compile(src,str(V/'code/long_incremental.py'),'exec'));out=E
 from wide_stream_window import window
 from wide_stream_alternate import alternate
 choice=np.array(read(O/'initial.json')['metrics']['fixed_acp']);pool=np.ascontiguousarray(raw[:N0+k*n]);pool=pool[np.random.default_rng(20261026+k).permutation(len(pool))];state=OwnerState(g,26);path,sec=run_window(state,pool,out/f'parallel_full_profile_{k:02d}');del pool;data=np.ascontiguousarray(raw[N0+(k-1)*n:N0+k*n]);before=fingerprint(state);path,s2=run_window(state,data,out/f'parallel_full_rebuild_{k:02d}');check_old(state,before);met,_=count(state,data,path,choice);save(out/f'full_result_{k:02d}.json',dict(policy='full_rebuild',window=k,metrics=met,seconds=sec+s2));print('DONE',k,flush=True);sys.exit(0)
# Explicit process IDs are needed only when taking over live serial jobs.
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--initial-pid',type=int);parser.add_argument('--extension-pid',type=int);args=parser.parse_args()
if (E/'complete.json').exists():print('Already complete; no results overwritten');sys.exit(0)
def stop_matching(pid,filename):
 if pid is None:return
 try:
  cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
  if not any(x.endswith(('/'+filename).encode()) for x in cmdline):raise RuntimeError(f'PID {pid} is not the expected {filename}')
  os.kill(pid,signal.SIGTERM)
 except (FileNotFoundError,ProcessLookupError):pass
# Stop the redundant initial checkpoint sequence once its first complete result is durable.
stopped=False
while not (E/'final_frozen_state/manifest.json').exists():
 if not stopped:
  try:
   d=json.loads((O/'summary.json').read_text())
   if any(r['policy']=='full_rebuild' and r['window']==1 for r in d['results']):stop_matching(args.initial_pid,'long_incremental.py');stopped=True;(O/'checkpoint_sequence_shortened.json').write_text(json.dumps(dict(reason='replace redundant W8-W32 full rebuilds with stabilized-tail full rebuilds',retained_full_checkpoint=1)))
  except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):pass
 time.sleep(5)
try:stop_matching(args.extension_pid,'extend_incremental.py')
except ProcessLookupError:pass
d=json.loads((E/'summary.json').read_text());K=d['windows'];queue=list(range(K-7,K+1));active=[];finished=[]
while queue or active:
 while queue and len(active)<4:
  k=queue.pop(0);log=open(V/f'logs/parallel_full_{k:02d}.log','w');p=subprocess.Popen([sys.executable,__file__,'worker',str(k)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'8'});active.append((k,p,log));print('START',k,flush=True)
 for k,p,log in active[:]:
  code=p.poll()
  if code is not None:
   log.close();active.remove((k,p,log))
   if code:raise RuntimeError(f'full rebuild {k} failed: {code}')
   finished.append(k);print('DONE',k,flush=True)
 time.sleep(5)
# Merge only complete, audited results; retain every incremental/frozen window.
d=json.loads((E/'summary.json').read_text());d['results']=[r for r in d['results'] if r['policy']!='full_rebuild'];d['results'] += [json.loads((E/f'full_result_{k:02d}.json').read_text()) for k in sorted(finished)];d['results'] += [r for r in json.loads((O/'summary.json').read_text())['results'] if r['policy']=='full_rebuild'];d['full_checkpoints']=sorted({r['window'] for r in d['results'] if r['policy']=='full_rebuild'});(E/'summary.json').write_text(json.dumps(d,indent=2))
s=json.loads((E/'stability_checks.json').read_text())[-1];td={(r['policy'],r['window']):r['metrics']['vectors_per_8192'] for r in d['results']};ratios=[td['incremental',k]/td['full_rebuild',k] for k in range(K-7,K+1)];s.update(last8_incremental_over_full=ratios,last8_mean_ratio=sum(ratios)/8,last8_ratio_of_sums=sum(td['incremental',k] for k in range(K-7,K+1))/sum(td['full_rebuild',k] for k in range(K-7,K+1)),tail_start=K-7,tail_end=K);(E/'stability.json').write_text(json.dumps(s,indent=2));(E/'complete.json').write_text(json.dumps(dict(success=True,stable=s['stable'],parallel_workers=4,old_embeddings_moved=0)));print('COMPLETE',s,flush=True)
