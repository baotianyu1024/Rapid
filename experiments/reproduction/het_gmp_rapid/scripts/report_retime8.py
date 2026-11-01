"""Report fresh eight-rank timings; do not mislabel mixed stages as pure compute."""
import json,re
from pathlib import Path
CASE=Path(__file__).resolve().parents[1]/'results/timing8_rerun'
def read(p):return json.loads(p.read_text())
def main():
 assert read(CASE/'preprocess_complete.json')['success']
 names=['input_and_gmp_adapter','gmp_partition','gmp_materialize','rapid_hot','rapid_placement','rapid_schedule','rapid_owner_base','rapid_price_f2','rapid_price_b8','rapid_price_b32','rapid_guard_p5','rapid_final_materialize']
 times={n:read(CASE/(n+'.status.json')) for n in names}
 assert all(x['exit_code']==0 for x in times.values())
 data=read(CASE/'data/input.json');part=read(CASE/'gmp/partition.json')
 assert data['rows']==4194304 and data['ranks']==8 and data['global_batches']==64
 assert sum(read(CASE/'gmp/materialization.json')['rank_rows'])==4194304
 assert read(CASE/'rapid_guard_p5/complete.json')['all_records_inverse_verified']
 adapter=times['input_and_gmp_adapter']['seconds']-data['copy_seconds']
 gmp=adapter+times['gmp_partition']['seconds']+times['gmp_materialize']['seconds']
 rapid=sum(v['seconds'] for n,v in times.items() if n.startswith('rapid_'))
 log=(CASE/'rapid_schedule.log').read_text()
 schedule={k:sum(map(float,re.findall(k+r'=([\d.]+)s',log))) for k in ['parse','hits','balance','write']}
 owner=read(CASE/'rapid_owner_base/phase_times.json')
 joint={s:read(CASE/('rapid_'+s)/'refinement.json') for s in ['price_f2','price_b8','price_b32','guard_p5']}
 material=read(CASE/'rapid_guard_p5/materialization.json')
 gcore={'graph':part['graph_seconds'],'samples':sum(r['data_seconds'] for r in part['rounds']),'embeddings':sum(r['embedding_seconds'] for r in part['rounds'])}
 clean=read(CASE/'cleanup_manifest.json')
 report=dict(rapid_preprocess_seconds=rapid,gmp_preprocess_seconds=gmp,rapid_over_gmp=rapid/gmp,common_input_copy_seconds=data['copy_seconds'],stages=times,
 rapid_schedule_phases=schedule,rapid_owner_base_mixed_phases=owner,rapid_joint_phases={k:{f:v[f] for f in ['kernel_seconds','optimization_seconds','load_and_prepare_seconds','save_arrays_seconds','total_seconds']} for k,v in joint.items()},
 rapid_final_materialization=material,gmp_isolated_compute=gcore,gmp_priority_export_mixed_seconds=part['export_priority_seconds'],
 gmp_diagnostics_seconds=sum(r['diagnostic_seconds'] for r in part['rounds']),gmp_guard_seconds=part['guard_seconds'],
 cleanup_files=len(clean['files']),cleanup_bytes=clean['bytes'],
 scope='Fresh 4194304-row eight-rank generation; sequential single trials; normal OS cache; selected RAPID warm-start chain rebuilt, fresh caches. Includes wrapper validation; excludes setup, historical tuning and post-run checks.',
 pure_compute_caveat='Only explicitly timed compute regions are reported. Hot-ID/placement/owner cache and mirror export contain mixed compute and I/O, so no complete pure-compute total is claimed.')
 (CASE/'timing_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 labels={'rapid_hot':'热 ID 提取','rapid_placement':'初始 ID 放置','rapid_schedule':'命中统计、样本平衡及调度输出','rapid_owner_base':'初始 owner 优化、第一次重映射和校验','rapid_price_f2':'第一轮联合均衡（包含全新 profile 构建）','rapid_price_b8':'反向均衡 warm-start','rapid_price_b32':'强化反向与存储均衡','rapid_guard_p5':'保持均衡约束的最终细化','rapid_final_materialize':'最终映射、写出和完整校验','gmp_partition':'建图、10 轮分区、mirror 排序与导出','gmp_materialize':'按 rank 生成完整样本文件'}
 lines=['8-rank RAPID / GMP 全新生成计时','',
 '同一 Criteo day6 前 4,194,304 个样本；8 rank；local batch 8192；64 个全局 batch。',
 'CPU 离线生成，不是 GPU 训练。OMP=16，BLAS=1，调度 num_threads=1。',
 '两算法顺序执行，不复用旧分区、owner、profile 或 scheduled 文件。',
 'RAPID 使用当前精确加速 scheduler，以及 f2 -> b8 -> b32 -> guard_p5 固定依赖链。',
 '最终映射归档不压缩；此前已证明映射数组与训练文件内容不受压缩选项影响。',
 'GMP 原生 10 轮、store_rate=.01。正常文件缓存；删除数据不等于清空系统缓存。','',
 f'RAPID 全流程：{rapid:.3f} 秒 / {rapid/60:.3f} 分钟。',f'GMP 全流程：{gmp:.3f} 秒 / {gmp/60:.3f} 分钟。',f'RAPID/GMP：{rapid/gmp:.3f}。','',
 f'GMP 无损 ID 编码准备（扣除公共切片复制）：{adapter:.3f} 秒。']
 lines += [f'{n} | {labels[n]}：{times[n]["seconds"]:.3f} 秒。' for n in names if n in labels]
 lines += ['', '以下内部计时已包含在全流程中，不重复相加：',
 f'RAPID 命中统计：{schedule["hits"]:.3f} 秒；样本平衡：{schedule["balance"]:.3f} 秒；合计：{schedule["hits"]+schedule["balance"]:.3f} 秒。',
 f'RAPID 四阶段 owner C++ 内核合计：{sum(v["kernel_seconds"] for v in joint.values()):.3f} 秒。',
 f'GMP 建图：{gcore["graph"]:.3f} 秒；样本分区：{gcore["samples"]:.3f} 秒；embedding 分区：{gcore["embeddings"]:.3f} 秒；合计：{sum(gcore.values()):.3f} 秒。',
 f'GMP mirror 优先级/排序/导出混合阶段：{part["export_priority_seconds"]:.3f} 秒。',
 '上述核心计时不覆盖所有算法工作，不能视为同范围完整纯计算总耗时。',
 '最终 RAPID materialization 分项：'+json.dumps(material['timings'],ensure_ascii=False),
 '',f'公共输入切片复制 {data["copy_seconds"]:.3f} 秒单列，未计入任一算法；若从完整 day6 文件起算，两者各加同样的这段时间。',
 '计入当前脚本自带的校验和诊断；耗时分项保留在 timing_summary.json。各执行一次，非多次统计。',
 f'清理 {len(clean["files"])} 个可重建的旧文件，逻辑大小 {clean["bytes"]/1e9:.3f} GB。保留原始输入、最优方案依赖、映射和报告。']
 if (CASE/'postrun_verification.json').exists():
  verified=read(CASE/'postrun_verification.json');assert verified['success']
  report['postrun_verification']=verified
  (CASE/'timing_summary.json').write_text(json.dumps(report,indent=2)+'\n')
  lines += ['', '计时后复核（不计入生成耗时）：GMP 全部输出记录与源数据对应；RAPID 完整逆映射和独立通信计数通过。']
  for stage,matches in verified['rapid_historical_shard_matches'].items():
   lines.append(f'{stage} 与历史同配置输出逐字节一致：{sum(matches)}/8 个 rank。')
 (CASE/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
