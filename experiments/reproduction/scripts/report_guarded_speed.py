"""Publish verified communication refinements and separately scoped exact speed tests."""
import json,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/guarded_speed_comparison';out.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
prior=read(root/'results/joint_price_comparison/comparison.json');communication={};lines=['RAPID：保留均衡约束的通信细化与结果一致的代码加速','',
 '通信单位为 embedding/梯度向量个数；不是字节。数据为 Criteo day6 的同一批 4,194,304 个样本。',
 'RAPID 每 rank 每 batch 8192 个样本，无 mirror；GMP s=0，保留其原有 mirror。',
 '峰值均值取每步最大接收量后跨步平均。GMP 均值诊断使用完整 batch；总量包含全部尾批和 5 个 epoch。','']
for g,seed in [(8,'criteo_day6_8_price_b32'),(32,'criteo_day6_32_price_warm')]:
 case=root/f'results/criteo_day6_{g}_guard_p5';new=read(case/'refinement.json');old=read(root/f'results/{seed}/refinement.json');verify=read(case/'complete.json');assert verify['independent_counts_verified'] and verify['all_records_inverse_verified']
 x=np.load(case/'cps_counts.npz');y=np.load(root/f'results/{seed}/cps_counts.npz')
 for k in ['receive','send']:
  a=x[k].sum(1);b=y[k].sum(1);assert np.all(a.max(1)<=b.max(1));assert (a.max(1)/a.mean(1)).max()<=(b.max(1)/b.mean(1)).max()+1e-12
 assert min(new['after_storage'])>=min(old['after_storage']) and max(new['after_storage'])<=max(old['after_storage'])
 total=int((x['receive'].sum()+x['send'].sum())*5);oldtotal=int((y['receive'].sum()+y['send'].sum())*5)
 communication[g]=dict(previous=old,current=new,gmp=prior[str(g)]['gmp'],guards_verified=True,all_records_verified=True,total_vectors_5epochs=total,traffic_change_vs_previous=total/oldtotal-1)
 lines += [f'{g} rank：', '方案 | 前向峰值均值 | 反向峰值均值 | 最差前向 max/mean | 最差反向 max/mean | owner 最大/平均']
 for name,j in [('上一版',old),('本次细化',new)]:
  f,b=j['after_forward'],j['after_backward'];s=np.array(j['after_storage']);lines.append(f"{name} | {f['mean_batch_peak']:.3f} | {b['mean_batch_peak']:.3f} | {f['max_batch_peak_over_mean']:.6f} | {b['max_batch_peak_over_mean']:.6f} | {s.max()/s.mean():.6f}")
 gm=prior[str(g)]['gmp'];f,b=gm['forward'],gm['backward'];lines.append(f"GMP | {f['mean_batch_peak']:.3f} | {b['mean_batch_peak']:.3f} | {f['max_batch_peak_over_mean']:.6f} | {b['max_batch_peak_over_mean']:.6f} | {gm['primary_storage']['max_over_mean']:.6f}")
 lines += [f'全部 5 epoch 通信：上一版 {oldtotal}，本次 {total}，GMP {gm["total_vectors_5epochs"]}。',f'相对上一版总通信变化：{100*(total/oldtotal-1):+.4f}%。','']
audit=read(root/'results/speed_exact/historical_time_audit.json');owner=read(root/'results/speed_exact/owner_kernel_benchmark.json');scheduler=read(root/'results/speed_exact/scheduler_benchmark.json');maps=read(root/'results/speed_exact/materialization_benchmark.json');vector=read(root/'results/speed_exact_vector/owner_kernel_benchmark.json')
speed=dict(owner=owner,vector=vector,scheduler=scheduler,materialization=maps)
lines += ['速度测试与通信优化分开验证：以下 A/B 使用相同起点、参数和输出目标，不通过减少搜索获得加速。','',
 'owner 内核：全部 55,874,235 个 owner 和每 batch/rank/table 接收计数完全一致，输入已加载到内存。']
for label,data in [('原内核与第一版加速',owner),('原内核与候选并列计算',vector)]:
 groups={}
 for row in data['results']:groups.setdefault(row['version'],[]).append(row['kernel_wall_seconds'])
 lines.append(label+'：'+', '.join(f'{k}={np.mean(v):.4f}s（{len(v)}次）' for k,v in groups.items()))
lines.append('候选并列计算版本结果一致但更慢，已淘汰；默认使用第一版加速内核。')
lines.append('调度实现额外通过 60 组 weighted/ratio 输出差分，以及独立双向/三向交换计数检查。')
a,b=scheduler['results'];lines += [f"样本调度（2 个完整 32-rank batch，524,288 行）：平衡核心 {a['phases']['balance']:.4f}s -> {b['phases']['balance']:.4f}s，{a['phases']['balance']/b['phases']['balance']:.3f} 倍。",f"含初始化/写出的调度阶段：{a['wall_seconds']:.4f}s -> {b['wall_seconds']:.4f}s。所有输出字节与历史基线一致。",'该测量不能直接称为完整 16-batch 的加速比例；计数表逻辑另通过 100 万次与 unordered_map 的差分操作测试。','']
for m in maps:
 lines.append(f"全量映射及校验 compressed={m['compressed']}：总墙钟 {m['external_wall_seconds']:.4f}s；分项 {m['phases']['timings']}；映射归档文件 {m['map_archive_bytes']} bytes。")
lines += ['压缩/不压缩两个版本的全部映射数组及所有 4,194,304 行数据字节均与上一版完全相同。未压缩文件更大。','',
 '历史预处理计时审计（32 rank，顺序执行）：',
 f"RAPID 可单独确认的命中统计＋样本平衡计算：{audit['rapid_isolated_assignment_core_seconds']:.4f}s。",
 f"GMP 可单独确认的建图＋10轮样本/embedding分区计算：{audit['gmp_isolated_core_seconds']:.4f}s。",
 f"GMP priority/mirror 排序与导出混合阶段另有 {audit['gmp_priority_and_export_mixed_seconds']:.4f}s。",
 '以上不是完全同范围的全算法纯计算总时间。RAPID 还需热度、METIS、放置、owner 优化；GMP 还需编码和 mirror 优先级/排序。',
 '即使仅计算已单独计时的 RAPID 调度核心，其历史耗时也超过 GMP 历史完整预处理的 1564.26s。不能把 RAPID 较慢归因于最终写数据。',
 '历史 RAPID 全预处理 2851.16s，最终记录映射/重写 409.32s（含 ID 转换计算，不是纯磁盘 I/O）。',
 '旧 materialization.json 的总 seconds 字段有循环变量覆盖错误，本次已修正并新增分项计时；历史外部进程计时不受影响。',
 '最新 owner 优化以此前 v7 结果为起点，所测内核时间不是从原始数据开始的完整 RAPID 计算时间。',
 '没有测 GPU 训练吞吐、精度或留出数据，不宣称全面超过 GMP。8-rank 前向和反向最差均衡比仍有差距。','',
 '其他尝试：按真实去重通信增量交换样本，在 8 个 batch 子集上降低约 0.14% 总通信，未作为全量结果采用。',
 '从原始 owner 直接做 hot-first 初始化仍未达到当前均衡水平，未替换已验证的 warm-start 流程。']
(out/'comparison.json').write_text(json.dumps(dict(communication=communication,speed=speed,historical_time_audit=audit),indent=2));(out/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
