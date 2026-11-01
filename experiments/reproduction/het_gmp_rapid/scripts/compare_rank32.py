"""Compare selected RAPID output against a newly partitioned 32-rank HET replay."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'results/rank32_best'
RAPID = ROOT.parent/'results/criteo_day6_32_owner_balanced_16b'
P, B, D, EPOCHS = 32, 8192, 128, 5

def read(p):
    return json.loads(p.read_text())

def stats(a):
    peaks = a.max(axis=1)
    return dict(steps=len(a), mean_vectors=float(a.mean()),
        mean_batch_max_vectors=float(peaks.mean()), absolute_max_vectors=int(a.max()),
        max_rank_mean_vectors=float(a.mean(axis=0).max()), rank_mean_vectors=a.mean(axis=0).tolist(),
        rank_absolute_max_vectors=a.max(axis=0).tolist(), total_vectors=int(a.sum()))

def main():
    cfg=read(RAPID/'config.json')
    assert (cfg['ranks'],cfg['batch_size'],cfg['rows'])==(P,B,4194304)
    recounted=read(CASE/'rapid_recount.json')
    recorded=read(RAPID/'cps_metrics.json')
    with np.load(CASE/'rapid_recount.npz') as new, np.load(RAPID/'cps_counts.npz') as old:
        for k in ['receive','send','unique','hits']:
            assert np.array_equal(new[k],old[k]), f'Recount differs: {k}'
        recv=new['receive']; send=new['send']
        assert recv.shape==(16,26,32)
        # Original B*P and remote B*(P-1) ACP thresholds both choose all Unique.
        assert np.all(recv.mean(axis=(0,2)) < B*(P-1))
        rf=np.tile(recv.sum(axis=1),(EPOCHS,1))
        rb=np.tile(send.sum(axis=1),(EPOCHS,1))
    assert stats(rf)['mean_vectors']==recorded['acp_receive']['mean']
    assert stats(rf)['mean_batch_max_vectors']==116246.0
    hm=read(CASE/'traffic/het_gmp_s0.json')
    assert hm['rows']==cfg['rows'] and hm['epochs']==EPOCHS
    with np.load(CASE/'traffic/het_gmp_s0.npz') as h:
        hf=h['forward'].sum(axis=2)
        hb=h['backward'].sum(axis=1)
    assert hf.shape==(hm['steps'],P)
    assert hf.sum()==hm['forward_receive_vectors']['total']
    assert hb.sum()==hm['backward_receive_vectors']['total']
    assert hm['pending_updates']==0
    full_het=np.arange(hm['steps'])%hm['steps_per_epoch'] < min(hm['rank_rows'])//B
    results={}
    for name, f, b, full, per_epoch in [
        ('rapid_best',rf,rb,np.ones(len(rf),bool),16),
        ('het_gmp_s0',hf,hb,full_het,hm['steps_per_epoch'])]:
        results[name]={}
        for scope, mask in [('all_steps',np.ones(len(f),bool)),('all_ranks_full_batch',full),
                            ('first_epoch',np.arange(len(f))<per_epoch)]:
            results[name][scope]={label:stats(a[mask]) for label,a in
                [('forward',f),('backward',b),('forward_plus_backward',f+b)]}
        np.savetxt(CASE/f'{name}_receive_vectors.csv',f,fmt='%d',delimiter=',',
            header=','.join(f'rank_{r}' for r in range(P)),comments='')
    input_meta=read(ROOT/'results/main64/data/input.json')
    sources=[RAPID/'config.json',RAPID/'cps_counts.npz',RAPID/'owner_refinement.json',
        CASE/'het/partition.json',CASE/'het/owner.npy',CASE/'het/assignment.npy',CASE/'traffic/het_gmp_s0.npz']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report=dict(ranks=P,local_batch=B,embedding_dim=D,dtype='FP32',epochs=EPOCHS,
        rows=cfg['rows'],input_sha256=input_meta['input_sha256'],rapid_case=str(RAPID),
        rapid_recount_matches_saved=True,gmp_new_partition=True,gmp_bound=0,gmp_store_rate=.01,
        gmp_topology='homogeneous logical ranks: off-diagonal cost 0.1, diagonal 0; no physical topology claim',
        scope='remote received embedding/gradient vectors; excludes self, metadata, dense allreduce',
        gmp_rank_rows=hm['rank_rows'],sources_sha256=hashes,results=results)
    (CASE/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['32-rank 最优 RAPID 与 HET-GMP 接收通信量对比','',
        '数据：Criteo day6 同一前 4,194,304 行；32 rank，每卡 batch 8192，128 维 FP32，5 epoch。',
        'RAPID 使用先前选定的 owner_balanced 优化版，而非原始 RAPID；本次逐条重新扫描输出数据，',
        'receive/send/unique/hits 计数与既有保存结果完全一致，26 表均选择 Unique。',
        'GMP 本次重新执行原生 32 分区、10 轮 refinement、1% mirror、s=0 的 CPU 协议回放。',
        '采用等成本逻辑 rank 拓扑；未指定实际多机链路。未跑端到端训练。','',
        '前向远端接收 embedding 向量数（即实际取回向量对应的 ID 数，不是查询元数据 ID 数）：',
        '方案/范围                     步数      rank×step均值       每步32rank最大值的均值       绝对最大值']
    for scope in ['all_steps','all_ranks_full_batch','first_epoch']:
        for name in results:
            s=results[name][scope]['forward']
            lines.append(f"{name}/{scope}: {s['steps']}  {s['mean_vectors']:.6f}  {s['mean_batch_max_vectors']:.6f}  {s['absolute_max_vectors']}")
    lines += ['', '所有通信量只以 emb 向量个数表示，不换算字节大小。',
        'mean = 所有 rank、所有步骤接收数的平均；mean_batch_max = 先每步取32rank最大值再平均；',
        'absolute_max = 所有步骤和rank的单次最大值。不能把 max(rank均值) 与 mean_batch_max 混用。',
        'GMP 不丢弃/重复样本，允许不等长尾批；完整batch口径只保留每张卡均有8192样本的步骤，',
        '用于排除尾批对均值的稀释；不是相同样本集合的重新总量对比。',
        'first_epoch 单列冷启动；主结果为5轮连续回放，GMP版本状态跨epoch保留。',
        'GMP 原生并行启发式可能存在运行间差异；本次保存的分区确定本次结果。',
        'RAPID 最优版本曾在同一trace上优化和筛选；不是原始默认算法或未见数据泛化结果。',
        '逐rank逐step计数见 *_receive_vectors.csv；反向和前反向合计、每rank均值见 comparison.json。',
        '', '复核：python scripts/compare_rank32.py（het-gmp-rapid 环境）。']
    (CASE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
