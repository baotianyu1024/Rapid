"""Report wall time of the freshly rerun, fixed 32-rank preprocessing workflows."""
import json
from pathlib import Path

CASE=Path(__file__).resolve().parents[1]/'results/timing32_rerun'
def read(path):return json.loads(path.read_text())

def main():
    assert read(CASE/'preprocess_complete.json')['success']
    names=['input_and_gmp_adapter','gmp_partition','gmp_materialize','rapid_hot','rapid_metis_seed',
           'rapid_placement','rapid_schedule','rapid_owner_and_acp']
    times={n:read(CASE/f'{n}.status.json') for n in names}
    assert all(v['exit_code']==0 for v in times.values())
    data=read(CASE/'data/input.json')
    phases=read(CASE/'rapid_best/phase_times.json')
    part=read(CASE/'gmp/partition.json')
    material=read(CASE/'gmp/materialization.json')
    assert sum(material['rank_rows'])==4194304 and len(material['rank_rows'])==32
    assert data['ranks']==32 and data['global_batches']==16
    adapter=times['input_and_gmp_adapter']['seconds']-data['copy_seconds']
    gmp=adapter+times['gmp_partition']['seconds']+times['gmp_materialize']['seconds']
    rapid=sum(times[n]['seconds'] for n in names if n.startswith('rapid_'))
    metrics=read(CASE/'rapid_best/cps_metrics.json')
    rows=[('RAPID','热 ID 提取',times['rapid_hot']['seconds']),
          ('RAPID','METIS 初始化（含建图与写出）',times['rapid_metis_seed']['seconds']),
          ('RAPID','ID 放置',times['rapid_placement']['seconds']),
          ('RAPID','优化样本调度及写出',times['rapid_schedule']['seconds']),
          ('RAPID','最终 owner 优化、重写及 ACP profile',times['rapid_owner_and_acp']['seconds']),
          ('GMP','无损 ID 编码准备（扣除公共输入切片复制）',adapter),
          ('GMP','建图、10 轮分区、mirror 排序与导出',times['gmp_partition']['seconds']),
          ('GMP','按 rank 写出样本数据',times['gmp_materialize']['seconds'])]
    cleanup=read(CASE/'cleanup_manifest.json')
    report=dict(rapid_preprocess_seconds=rapid,gmp_preprocess_seconds=gmp,
        rapid_over_gmp=rapid/gmp,common_input_copy_seconds=data['copy_seconds'],
        cleanup_files=len(cleanup['files']),cleanup_logical_bytes=cleanup['total_bytes'],
        stages=times,rapid_owner_phase_times=phases,
        gmp_partition_guard_seconds=part['guard_seconds'],
        gmp_partition_diagnostics_seconds=sum(r['diagnostic_seconds'] for r in part['rounds']),
        rapid_forward_receive_emb=metrics['acp_receive'],
        scope='fresh fixed-configuration preprocessing including data output and required profiles; normal OS cache; sequential single trials; excludes historical parameter search and post-run communication replay',
        note='Includes runtime validation/diagnostics of the executed wrappers; not a pure algorithm-only timing or a cold-cache experiment')
    (CASE/'timing_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['32-rank 最优 RAPID 与 HET-GMP 重新计时','',
        '输入：相同 Criteo day6 前 4,194,304 行，32 rank，每 rank batch 8192。',
        '执行：清理旧生成数据后，从原始输入重新进行预处理；两算法顺序执行，不相互并发。',
        'RAPID 为之前最优 owner_balanced 的固定参数复测，不重新搜索超参。',
        'GMP 为原生 10 轮分区、1% mirror；其 s=0 通信协议不影响离线预处理。',
        'OMP=16，OPENBLAS/MKL=1；RAPID schedule num_threads=1，原设置保持不变。',
        '使用正常文件缓存，未执行系统级 drop_caches；不是冷盘性能测试。','',
        f'RAPID 预处理合计：{rapid:.3f} 秒，{rapid/60:.3f} 分钟。',
        f'GMP 预处理合计：{gmp:.3f} 秒，{gmp/60:.3f} 分钟。',
        f'本次 RAPID/GMP 耗时比：{rapid/gmp:.3f}。','',
        '阶段（墙钟秒）：']
    lines += [f'{algo} | {stage}: {seconds:.3f}' for algo,stage,seconds in rows]
    lines += ['', 'RAPID 最后一步内部计时（不是额外再加到总时间）：']
    lines += [f'{name}: {seconds:.3f}' for name,seconds in phases.items()]
    lines += ['',f"公共输入切片复制：{data['copy_seconds']:.3f} 秒，未算入任一算法。",
        'GMP 包含 ID 编码适配和等价样本写出；RAPID 包含热度统计、METIS 图初始化、',
        '放置、最优参数调度、owner 优化、可逆重映射和最终 ACP profile。',
        '运行时检查和诊断包含在进程墙钟内，细分见 timing_summary.json。',
        '未计入安装编译、历史超参搜索、此前 owner 参数核对，以及后续通信量回放。',
        'ACP profile 与 owner 验证共用一次逐表计数，已计入 RAPID 最后一步，未把它隐藏。',
        '这是同机各一次执行，不能当作多次测量置信区间或算法理论复杂度。',
        '', f"清理：{len(cleanup['files'])} 个旧 scheduled 数据文件；逐文件清单见 cleanup_manifest.json。",
        '原始数据、所选最优结果及其依赖、所有原报告/统计/可逆映射保留。',
        'RAPID placement/hot 源码与旧快照哈希一致；scheduler 使用哈希精确匹配的',
        'generate_local_dataset.pre_hybrid，而不是后来修改的当前二进制。',
        'owner 参数先在旧 trace 上复核，接收矩阵和存储计数与旧最优结果完全一致；',
        '本次输入和所有中间产物重新生成。原生并行启发式可能导致产物数值变化。',
        '', 'RAPID 本次前向接收统计（只以 emb 个数表示）：',
        f"rank×step平均={metrics['acp_receive']['mean']:.6f}；每步最大值均值={metrics['acp_receive']['mean_batch_peak']:.6f}；绝对峰值={metrics['acp_receive']['absolute_peak']:.0f}。",
        '全部阶段的精确命令、GNU time CPU/RSS、退出码和日志保存在本目录；source_sha256.json 保存源码/二进制哈希。']
    (CASE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
