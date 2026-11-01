"""GMP-inspired joint forward/backward/storage refinement of a fixed RAPID trace.

Exact batch/rank ID presence, no mirrors, unchanged sample assignment. The
cache is reusable across policy experiments. Final IDs are compact per owner
and table; independently rescan materialized records before declaring success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
from measure import measure, summarize


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def deltas(profile, weight, source, dest, batches):
    powers = np.uint64(1) << np.arange(batches, dtype=np.uint64)
    local_s = ((profile[:, source, None] & powers) != 0).astype(np.int64)
    local_d = ((profile[:, dest, None] & powers) != 0).astype(np.int64)
    return local_s, local_d, weight.astype(np.int64) - local_s, weight.astype(np.int64) - local_d


def best_swap(tables, send_detail, send, cap, batches):
    """Swap same-table IDs with equal presence on both involved ranks.

    Each rank's forward receive counts and per-table owner cardinalities remain
    identical. Different remote fanout redistributes owner service load.
    """
    assert batches <= 32
    excess = np.maximum(send/cap[:,None]-1,0)
    best, best_gain = None, 1e-12
    for source in np.argsort(-excess.max(axis=0))[:3]:
        if not excess[:,source].any(): continue
        for dest in np.argsort((send/cap[:,None]).max(axis=0))[:12]:
            if source==dest or np.any(send[:,dest]>=cap): continue
            for t in np.argsort(-send_detail[:,:,source].sum(axis=0))[:8]:
                table=tables[t];owners=table['owners'];profiles=table['profiles'];mass=table['mass']
                hi=np.flatnonzero(owners==source);lo=np.flatnonzero(owners==dest)
                if not len(hi) or not len(lo):continue
                if len(hi)>32768:
                    hi=hi[np.argpartition(-mass[hi],32767)[:32768]]
                hi=hi[np.argsort(-mass[hi],kind='stable')]
                hc=profiles[hi,source].astype(np.uint64)|(profiles[hi,dest].astype(np.uint64)<<np.uint64(batches))
                lc=profiles[lo,source].astype(np.uint64)|(profiles[lo,dest].astype(np.uint64)<<np.uint64(batches))
                # Include codes carried by the hottest IDs, plus frequent codes.
                codes,cnt=np.unique(hc,return_counts=True)
                codes=np.unique(np.concatenate((hc[:32],codes[np.argsort(-cnt)[:16]])))
                hot,low=[],[]
                for code in codes:
                    h=hi[hc==code];l=lo[lc==code]
                    if not len(l):continue
                    l=l[np.argsort(mass[l],kind='stable')]
                    n=min(len(h),len(l),16384);h=h[:n];l=l[:n]
                    valid=mass[h]>mass[l]
                    hot.extend(h[valid]);low.extend(l[valid])
                if not hot:continue
                hot=np.array(hot,dtype=np.int64);low=np.array(low,dtype=np.int64)
                order=np.argsort(-(mass[hot]-mass[low]),kind='stable')
                hot,low=hot[order],low[order]
                change=table['weights'][hot].astype(np.int64)-table['weights'][low].astype(np.int64)
                valid=((send[:,dest]+change<=cap)&(change>=0)).all(axis=1)
                hot,low,change=hot[valid],low[valid],change[valid]
                if not len(hot):continue
                delta=np.cumsum(change,axis=0)
                ns=send[:,source]-delta;nd=send[:,dest]+delta
                feasible=((ns>=0)&(nd<=cap)).all(axis=1)
                gain=(excess[:,source]**2-np.maximum(ns/cap-1,0)**2).sum(axis=1)
                gain[~feasible]=-np.inf;k=int(gain.argmax())
                if gain[k]>best_gain:
                    best_gain=float(gain[k]);best=(int(t),int(source),int(dest),hot[:k+1],low[:k+1],delta[k])
    return best


def build_cache(case, cache, cfg):
    cache.mkdir(parents=True, exist_ok=True)
    identity = dict(case=str(case), config=cfg,
                    counts_sha256=hashlib.sha256((case/'cps_counts.npz').read_bytes()).hexdigest())
    if (cache/'identity.json').exists():
        assert json.loads((cache/'identity.json').read_text()) == identity
    else:
        save(cache/'identity.json', identity)
    g, nb, b = cfg['ranks'], cfg['batches'], cfg['batch_size']
    assert nb <= 64
    dt = np.uint16 if nb <= 16 else np.uint64
    width = 1 + cfg['dense'] + sum(cfg['table_lengths'])
    arrays = [np.memmap(case/'scheduled'/f'dataset_{r}.bin', dtype='<u4', mode='r').reshape(-1, width) for r in range(g)]
    offsets = np.cumsum([1+cfg['dense']] + cfg['table_lengths'])
    for t in range(len(cfg['table_lengths'])):
        if (cache/f'{t}.done').exists():
            continue
        lists = [np.unique(arrays[r][step*b:(step+1)*b, offsets[t]:offsets[t+1]])
                 for step in range(nb) for r in range(g)]
        ids = np.unique(np.concatenate(lists))
        profiles = np.zeros((len(ids), g), dtype=dt)
        weights = np.zeros((len(ids), nb), dtype=np.uint16)
        for q, values in enumerate(lists):
            pos = np.searchsorted(ids, values)
            profiles[pos, q % g] |= dt(1 << (q // g))
            weights[pos, q // g] += 1
        np.save(cache/f'{t}.ids.npy', ids)
        np.save(cache/f'{t}.profiles.npy', profiles)
        np.save(cache/f'{t}.weights.npy', weights)
        (cache/f'{t}.done').touch()
        print('profile table', t, 'IDs', len(ids), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--target', type=float, default=1.18)
    p.add_argument('--forward-target', type=float, default=1.20)
    p.add_argument('--forward-headroom', type=float, default=0.,
                   help='Allow this fractional increase in baseline-derived absolute forward cap; ratio cap still applies')
    p.add_argument('--backward-peak-growth', type=float, default=0.,
                   help='In storage-only mode, permit this growth over the resumed per-batch backward peak')
    p.add_argument('--storage-target', type=float, default=1.01)
    p.add_argument('--traffic-budget', type=float, default=.05)
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--resume-from', help='Initialize from a prior predicted placement on this same base case')
    p.add_argument('--swaps-first', action='store_true')
    p.add_argument('--storage-only', action='store_true',
                   help='Finish storage balancing without increasing existing backward peaks')
    p.add_argument('--materialize', action='store_true')
    p.add_argument('--uncompressed-maps',action='store_true',help='Store identical mapping arrays without ZIP compression')
    p.add_argument('--materialize-existing', action='store_true',
                   help='Write and independently verify a previously optimized output')
    a = p.parse_args()
    if not (a.target>=1 and a.forward_target>=1 and 1<=a.storage_target<2 and a.traffic_budget>=0 and a.rounds>=0
            and a.forward_headroom>=0 and a.backward_peak_growth>=0):
        p.error('Invalid balance limits or search budget')
    case, cache, out = map(lambda s: Path(s).resolve(), [a.case, a.cache, a.output])
    if a.materialize_existing:
        report = json.loads((out/'refinement.json').read_text())
        assert report['base_case'] == str(case)
        identity = json.loads((cache/'identity.json').read_text())
        assert identity['case'] == str(case)
        assert identity['counts_sha256'] == hashlib.sha256((case/'cps_counts.npz').read_bytes()).hexdigest()
        cfg = json.loads((case/'config.json').read_text())
        tables = [dict(ids=np.load(cache/f'{t}.ids.npy',mmap_mode='r'),
                       owners=np.load(out/f'{t}.owners.npy')) for t in range(len(cfg['table_lengths']))]
        predicted = np.load(out/'predicted_counts.npz')
        materialize(case, cache, out, cfg, tables, predicted['receive'], predicted['send'], compressed=not a.uncompressed_maps)
        return
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    stopped = [False]
    def request_stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    cfg = json.loads((case/'config.json').read_text())
    g, nb, b = cfg['ranks'], cfg['batches'], cfg['batch_size']
    base = np.load(case/'cps_counts.npz')
    assert np.all(base['receive'].mean(axis=(0, 2)) < b*(g-1)), 'Currently supports all-Unique traces'
    build_cache(case, cache, cfg)
    profile_seconds = time.perf_counter()-start
    tables = []
    storage = np.zeros(g, dtype=np.int64)
    resume = Path(a.resume_from).resolve() if a.resume_from else None
    if resume:
        assert json.loads((resume/'refinement.json').read_text())['base_case']==str(case)
    for t in range(len(cfg['table_lengths'])):
        ids = np.load(cache/f'{t}.ids.npy', mmap_mode='r')
        owners = np.load(resume/f'{t}.owners.npy') if resume else (ids % g).astype(np.int16)
        weights = np.load(cache/f'{t}.weights.npy', mmap_mode='r')
        tables.append(dict(ids=ids, owners=owners, weights=weights,
                           profiles=np.load(cache/f'{t}.profiles.npy', mmap_mode='r'),
                           mass=weights.sum(axis=1, dtype=np.int32)))
        storage += np.bincount(owners, minlength=g)
    before_storage = sum((np.bincount(table['ids']%g,minlength=g) for table in tables),np.zeros(g,dtype=np.int64))
    initial = np.load(resume/'predicted_counts.npz') if resume else base
    sd, rd = initial['send'].copy(), initial['receive'].copy()
    send, recv = sd.sum(axis=1), rd.sum(axis=1)
    total0 = base['receive'].sum(axis=(1,2))
    mean0 = total0/g
    # Absolute caps prevent improving ratios by artificially inflating traffic.
    forward_cap = np.maximum(recv.max(axis=1), mean0*a.forward_target*(1+a.forward_headroom))
    backward_cap = mean0*a.target
    if a.storage_only:
        backward_cap = send.max(axis=1).astype(float)*(1+a.backward_peak_growth)
    upper = int(np.ceil(storage.mean()*a.storage_target))
    lower = int(np.floor(storage.mean()*(2-a.storage_target)))
    history = []
    opt_start = time.perf_counter()
    if a.swaps_first:
        for iteration in range(60):
            if stopped[0]:break
            result=best_swap(tables,sd,send,backward_cap,nb)
            if result is None:break
            t,s,d,hi,lo,delta=result
            tables[t]['owners'][hi]=d;tables[t]['owners'][lo]=s
            send[:,s]-=delta;send[:,d]+=delta
            sd[:,t,s]-=delta;sd[:,t,d]+=delta
            h=dict(phase='swap',iteration=iteration,table=t,source=s,dest=d,pairs=len(hi),
                   backward_ratio=float((send.max(axis=1)/send.mean(axis=1)).max()))
            history.append(h);print(json.dumps(h),flush=True)
    for iteration in range(a.rounds):
        if stopped[0]:break
        if np.all(send <= backward_cap[:, None]) and storage.max() <= upper and storage.min() >= lower:
            break
        # Overloaded owners are selected using both service pressure and storage.
        pressure = np.maximum((send/backward_cap[:, None]).max(axis=0)-1, 0)
        if a.storage_only:
            pressure[:] = 0
        space = np.maximum(storage/upper-1, 0)
        if not np.any(pressure) and storage.min() < lower:
            space = np.maximum(storage/storage.mean()-1, 0)
        sources = np.argsort(-(pressure + 4*space))[:(2 if a.storage_only else 3)]
        best, best_gain = None, 1e-12
        for source in sources:
            if pressure[source] == 0 and space[source] == 0:
                continue
            storage_mode = pressure[source] == 0
            ranks = np.argsort((send/backward_cap[:, None]).max(axis=0) + storage/storage.mean())
            if a.storage_only:
                ranks = np.argsort(storage)
            destinations = [int(d) for d in ranks if d != source and storage[d] < upper][:(6 if a.storage_only else 12)]
            candidates_t = np.argsort(-sd[:, :, source].sum(axis=0))[:8]
            if storage_mode:
                candidates_t = sorted(range(len(tables)), key=lambda t: -np.count_nonzero(tables[t]['owners']==source))[:5]
            for t in candidates_t:
                table = tables[t]
                hi = np.flatnonzero(table['owners'] == source)
                # Apply saturated forward-batch constraints BEFORE truncating
                # candidates. Otherwise hot but infeasible IDs hide useful IDs.
                blocked = np.flatnonzero(forward_cap-recv[:,source] < 1)
                if len(blocked) and len(hi):
                    mask = sum(1 << int(step) for step in blocked)
                    hi = hi[(table['profiles'][hi,source].astype(np.uint64)&np.uint64(mask))==0]
                if not len(hi):
                    continue
                # Bound scoring cost, with high-degree IDs for traffic and cold IDs for storage.
                mass = table['mass'][hi]
                key = mass if storage_mode else -mass
                if len(hi)>65536:
                    keep = np.argpartition(key, 65535)[:65536]
                    hi, key = hi[keep], key[keep]
                hi = hi[np.argsort(key, kind='stable')]
                for dest in destinations:
                    n = min(16384, upper-int(storage[dest]),
                            int(storage[source])-lower if storage_mode else int(storage[source]))
                    if n <= 0:
                        continue
                    # Prefer a new owner already accessing the ID; allow a bounded locality loss.
                    h = hi
                    prof = table['profiles'][h]
                    ls, ld, ds, dd = deltas(prof, table['weights'][h], source, dest, nb)
                    valid = ((recv[:,source]+ls <= forward_cap)&
                             (send[:,dest]+dd <= backward_cap)&
                             (recv.sum(axis=1)+ls-ld <= total0*(1+a.traffic_budget))).all(axis=1)
                    if not valid.any():
                        continue
                    h = h[valid]
                    ls, ld, ds, dd = [v[valid] for v in (ls,ld,ds,dd)]
                    loss = (ls-ld).sum(axis=1)
                    order = np.argsort(loss, kind='stable')
                    ix = order[:n]
                    h = h[ix]
                    ls, ld, ds, dd = [np.cumsum(v[ix], axis=0) for v in (ls, ld, ds, dd)]
                    rs, rdest = recv[:, source]+ls, recv[:, dest]-ld
                    ss, sdest = send[:, source]-ds, send[:, dest]+dd
                    new_total = recv.sum(axis=1)+ls-ld
                    other_peak = np.max(np.delete(recv,[source,dest],axis=1),axis=1) if g>2 else np.zeros(nb)
                    new_peak = np.maximum(np.maximum(rs,rdest),other_peak)
                    feasible = ((rs <= forward_cap)&(rdest >= 0)&(ss >= 0)&
                                (sdest <= backward_cap)&(new_total <= total0*(1+a.traffic_budget))&
                                (new_peak <= new_total/g*a.forward_target)).all(axis=1)
                    # Squared overflow puts weight on the worst batches, not just mean traffic.
                    old_service = np.maximum(send[:, source]/backward_cap-1, 0)**2
                    new_service = np.maximum(ss/backward_cap-1, 0)**2
                    ns = storage[source]-np.arange(1,len(h)+1)
                    dest_ns = storage[dest]+np.arange(1,len(h)+1)
                    old_space = (storage[source]/storage.mean()-1)**2+(storage[dest]/storage.mean()-1)**2
                    new_space = (ns/storage.mean()-1)**2+(dest_ns/storage.mean()-1)**2
                    gain = (old_service-new_service).sum(axis=1) + 100*(old_space-new_space)
                    gain -= .001*((new_total-recv.sum(axis=1))/total0).sum(axis=1)
                    gain[~feasible] = -np.inf
                    k = int(gain.argmax())
                    if gain[k] > best_gain:
                        best_gain = float(gain[k])
                        best = (int(t), int(source), dest, h[:k+1].copy(), ls[k], ld[k], ds[k], dd[k])
        if best is None:
            print('no feasible improving migration', iteration, flush=True)
            break
        t, source, dest, idx, ls, ld, ds, dd = best
        tables[t]['owners'][idx] = dest
        send[:,source] -= ds; send[:,dest] += dd
        recv[:,source] += ls; recv[:,dest] -= ld
        sd[:,t,source] -= ds; sd[:,t,dest] += dd
        rd[:,t,source] += ls; rd[:,t,dest] -= ld
        storage[source] -= len(idx); storage[dest] += len(idx)
        assert np.array_equal(send.sum(axis=1), recv.sum(axis=1))
        h = dict(iteration=iteration, table=t, source=source, dest=dest, moved=len(idx),
                 backward_ratio=float((send.max(axis=1)/send.mean(axis=1)).max()),
                 forward_ratio=float((recv.max(axis=1)/recv.mean(axis=1)).max()),
                 storage_ratio=float(storage.max()/storage.mean()),
                 traffic_change=float(recv.sum()/total0.sum()-1))
        history.append(h)
        print(json.dumps(h), flush=True)
    opt_seconds = time.perf_counter()-opt_start
    for t, table in enumerate(tables):
        np.save(out/f'{t}.owners.npy', table['owners'])
    np.savez_compressed(out/'predicted_counts.npz', receive=rd, send=sd)
    report = dict(base_case=str(case), parameters=vars(a), profile_seconds=profile_seconds,
                  optimization_seconds=opt_seconds, history=history,
                  before_storage=before_storage.tolist(), after_storage=storage.tolist(),
                  before_forward=summarize(base['receive'].sum(axis=1)), after_forward=summarize(recv),
                  before_backward=summarize(base['send'].sum(axis=1)), after_backward=summarize(send),
                  traffic_change=float(recv.sum()/total0.sum()-1),
                  stopped_early=stopped[0],
                  scope='same offline trace; samples unchanged; no replicas; observed-ID storage, not allocated GPU memory')
    save(out/'refinement.json', report)
    save(out/'config.json', dict(cfg, output=str(out), base_case=str(case), balanced_owner_refinement=True))
    if a.materialize:
        materialize(case, cache, out, cfg, tables, rd, sd, compressed=not a.uncompressed_maps)
    print('DONE', out, flush=True)


def materialize(case, cache, out, cfg, tables, rd, sd, compressed=True):
    materialize_started = time.perf_counter()
    g, nb, b = cfg['ranks'], cfg['batches'], cfg['batch_size']
    timings=dict(map_generation_compute=0.,map_archive_write=0.,input_copy=0.,record_remap_compute=0.,dataset_write=0.,inverse_verification=0.,traffic_verification=0.)
    maps, inverses = [], []
    for t, table in enumerate(tables):
        phase=time.perf_counter()
        ids, owners = table['ids'], table['owners']
        new = np.empty(len(ids), dtype=np.uint32)
        for r in range(g):
            ix = np.flatnonzero(owners==r)
            values = np.arange(len(ix), dtype=np.uint64)*g+r
            if len(values): assert values[-1] <= np.iinfo(np.uint32).max
            new[ix] = values
        assert len(np.unique(new)) == len(ids)
        timings['map_generation_compute']+=time.perf_counter()-phase
        phase=time.perf_counter()
        writer=np.savez_compressed if compressed else np.savez
        writer(out/f'table_{t}_remap.npz', old=ids, new=new, owner=owners)
        timings['map_archive_write']+=time.perf_counter()-phase
        phase=time.perf_counter()
        # Refined RAPID IDs are usually compact. A direct lookup avoids binary
        # searching every sparse occurrence; fall back for very sparse ranges.
        lut = None
        if len(ids) and int(ids[-1])+1 <= max(1000000,8*len(ids)):
            lut = np.full(int(ids[-1])+1,-1,dtype=np.int64)
            lut[ids] = new
        maps.append((ids,new,lut))
        inverse = np.zeros(int(new.max())+1, dtype=np.uint32) if len(new) else np.zeros(0,dtype=np.uint32)
        inverse[new] = ids
        inverses.append(inverse)
        timings['map_generation_compute']+=time.perf_counter()-phase
    width=1+cfg['dense']+sum(cfg['table_lengths'])
    offsets=np.cumsum([1+cfg['dense']]+cfg['table_lengths'])
    (out/'scheduled').mkdir()
    for r in range(g):
        source=np.memmap(case/'scheduled'/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width)
        assert len(source)==nb*b
        with (out/'scheduled'/f'dataset_{r}.bin').open('wb') as f:
            for start in range(0,len(source),32768):
                phase=time.perf_counter()
                records=source[start:start+32768].copy()
                timings['input_copy']+=time.perf_counter()-phase
                phase=time.perf_counter()
                for t,(old,new,lut) in enumerate(maps):
                    cols=records[:,offsets[t]:offsets[t+1]]
                    if lut is not None:
                        mapped=lut[cols]
                        assert np.all(mapped>=0), 'Input ID absent from profile cache'
                        cols[:]=mapped
                    else:
                        ix=np.searchsorted(old,cols)
                        assert np.array_equal(old[ix],cols)
                        cols[:]=new[ix]
                timings['record_remap_compute']+=time.perf_counter()-phase
                phase=time.perf_counter()
                records.tofile(f)
                timings['dataset_write']+=time.perf_counter()-phase
        print('materialized rank',r,flush=True)
        # Re-read actual on-disk output and reverse every ID mapping. This also
        # verifies labels, dense values, multiplicities, order and batch sizes.
        phase=time.perf_counter()
        written=np.memmap(out/'scheduled'/f'dataset_{r}.bin',dtype='<u4',mode='r').reshape(-1,width)
        assert written.shape == source.shape
        for start in range(0,len(source),32768):
            restored=written[start:start+32768].copy()
            for t,inverse in enumerate(inverses):
                cols=restored[:,offsets[t]:offsets[t+1]]
                cols[:]=inverse[cols]
            assert np.array_equal(restored,source[start:start+32768]), 'Inverse record verification failed'
        timings['inverse_verification']+=time.perf_counter()-phase
    phase=time.perf_counter()
    metrics,actual=measure([out/'scheduled'/f'dataset_{r}.bin' for r in range(g)],g,b,nb,cfg['table_lengths'],cfg['dense'],True)
    base=np.load(case/'cps_counts.npz')
    assert np.array_equal(actual['unique'],base['unique'])
    assert np.array_equal(actual['receive'],rd)
    assert np.array_equal(actual['send'],sd)
    output_cfg=json.loads((out/'config.json').read_text())
    assert output_cfg.get('allow_reduction_tables',False) or all(t['strategy']=='Unique' for t in metrics['per_table'])
    timings['traffic_verification']+=time.perf_counter()-phase
    np.savez_compressed(out/'cps_counts.npz',**actual)
    save(out/'cps_metrics.json',metrics)
    save(out/'materialization.json',dict(seconds=time.perf_counter()-materialize_started,timings=timings,compressed_maps=compressed,
        includes='compact remap, output write, complete inverse-record verification, independent traffic scan'))
    save(out/'code_sha256.json',{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save(out/'complete.json',dict(success=True,independent_counts_verified=True,
                                 all_records_inverse_verified=True))


if __name__=='__main__':
    main()
