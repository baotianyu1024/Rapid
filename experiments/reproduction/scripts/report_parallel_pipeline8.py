"""Full-pipeline acceleration report; parallel batch sums are NOT wall time."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];NEW=R/'results/speed_parallel8/run';OLD=R/'het_gmp_rapid/results/timing8_rerun'
def read(p):return json.loads(p.read_text())
def main():
 assert read(NEW/'verification.json')['success']
 old=read(OLD/'timing_summary.json');names=[n for n in old['stages'] if n.startswith('rapid_')]
 now={n:read(NEW/(n+'.status.json')) for n in names};assert all(x['exit_code']==0 for x in now.values())
 total=sum(v['seconds'] for v in now.values());baseline=old['rapid_preprocess_seconds'];gmp=old['gmp_preprocess_seconds']
 log=(NEW/'rapid_schedule.log').read_text();mapping=float(re.search(r'mapping_wall=([\d.]+)s',log)[1]);processing=float(re.search(r'processing_wall=([\d.]+)s',log)[1])
 owner=read(NEW/'rapid_owner_base/phase_times.json');before=read(OLD/'rapid_owner_base/phase_times.json')
 report=dict(old_rapid_seconds=baseline,new_rapid_seconds=total,rapid_speedup=baseline/total,gmp_previous_seconds=gmp,new_rapid_over_gmp=total/gmp,
 stages={n:dict(old_seconds=old['stages'][n]['seconds'],new_seconds=now[n]['seconds'],speedup=old['stages'][n]['seconds']/now[n]['seconds']) for n in names},
 parallel_schedule_mapping_wall_seconds=mapping,parallel_schedule_processing_wall_seconds=processing,
 owner_base_phase_comparison=dict(old=before,new=owner),verification=read(NEW/'verification.json'),
 caveats=['Parallel per-batch durations overlap: their sum is CPU-work-like elapsed accumulation, not end-to-end latency. Use process wall or processing_wall.',
 'GMP comparison uses the immediately preceding measured run, not another GMP rerun; same input, host, OMP16/BLAS1, normal caches.',
 'All RAPID stages regenerated; common input slice already exists and its copying is excluded exactly as prior timing.',
 'Uncompressed archives preserve mapping arrays and training bytes, but archive bytes and disk usage differ.',
 'Single sequential trials; not a confidence interval or cold-cache benchmark.'])
 (NEW/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['RAPID 八卡生成：batch 并行与精确查表加速','',
 '输入：4,194,304 个样本，8 rank，每 rank batch8192，共64个batch。',
 '新流程从热度统计、放置开始全部重做；不复用旧 owner、placement、profile 或 scheduled 文件。',
 '16 个独立 batch worker，每个 worker 1 个 OpenMP 线程；每 batch 的算法、搜索预算、随机种子和决策顺序不变。',
 '通过按 batch 固定文件偏移写入，保证输出按原始全局 batch 顺序排列。',
 '初始 owner 重映射改为直接 LUT 查表；稀疏大 ID 范围保留二分查找回退。映射归档不压缩。','',
 f'旧 RAPID：{baseline:.3f} 秒 / {baseline/60:.3f} 分钟。',f'新 RAPID：{total:.3f} 秒 / {total/60:.3f} 分钟。',f'完整流程加速：{baseline/total:.3f} 倍。',
 f'上一轮 GMP 实测：{gmp:.3f} 秒 / {gmp/60:.3f} 分钟；新 RAPID/GMP={total/gmp:.3f}。','', '阶段 | 旧秒 | 新秒 | 加速倍数']
 lines += [f'{n} | {old["stages"][n]["seconds"]:.3f} | {now[n]["seconds"]:.3f} | {report["stages"][n]["speedup"]:.3f}' for n in names]
 lines += ['',f'并行调度加载映射墙钟：{mapping:.3f} 秒；64batch处理墙钟：{processing:.3f} 秒。',
 '不能把并行 batch 的 balance 计时相加后当作用户等待时间。',
 '初始 owner 阶段分项：'+json.dumps(report['owner_base_phase_comparison'],ensure_ascii=False),
 '调度、中间 owner 结果、最终结果：共24个分片逐字节一致；两套映射共52张表的全部数组完全相同。',
 '完整逆映射和通信计数校验在生成流程内；额外历史字节/数组比较在计时之外。',
 'GMP 使用上一轮实测结果；正常缓存、各一次测量。公共输入切片复制未计入两次RAPID或GMP。']
 (NEW/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
