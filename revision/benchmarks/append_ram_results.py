#!/usr/bin/env python3
"""Append measured results to the written report after all runs complete."""
import json
from pathlib import Path

root = Path('/tmp/rapid_ram_fair')
datasets = ('kaggle', 'terabyte', 'taobao')
results = {d: json.loads((root / d / 'summary.json').read_text()) for d in datasets}
inputs = {d: json.loads((root / d / 'het_input_summary.json').read_text()) for d in datasets}
rows = [('reduction', 'HugeCTR 全 Reduction'), ('unique', 'HugeCTR 全 Unique'),
        ('native', 'HugeCTR Taobao 全 Unique'), ('acp_only', 'hash + ACP-only'),
        ('materialized', 'RAPID 预物化'), ('online', 'RAPID 在线'),
        ('torch_prefetch', 'TorchRec 预取'), ('torch_gpu', 'TorchRec GPU 预放（诊断）'),
        ('het', 'GMP 普通 host 输入')]
lines = ['\n## 三数据集稳态吞吐\n',
         '单位 samples/s，三轮吞吐算术平均；GPU 预放仅为诊断。具体逐轮值和测量区间 I/O 在归档 JSON 中。\n',
         '| 配置 | Kaggle | Terabyte | Taobao |', '| --- | ---: | ---: | ---: |']
for key, name in rows:
    vals = [f"{results[d][key]['mean_samples_per_second']:,.0f}" if key in results[d] else '—' for d in datasets]
    lines.append('| ' + ' | '.join([name] + vals) + ' |')
lines += ['\n各配置逐轮 ms/步：\n', '| 配置 | Kaggle | Terabyte | Taobao |', '| --- | --- | --- | --- |']
for key, name in rows:
    vals = [' / '.join(f'{x*1000:.2f}' for x in results[d][key]['step_seconds']) if key in results[d] else '—' for d in datasets]
    lines.append('| ' + ' | '.join([name] + vals) + ' |')
lines += ['\n## GMP 输入三种模式对照\n',
          '三轮 ms/步；GPU 模式只预放全部稀疏 ID，dense/label 仍在 host。\n',
          '| 数据 | 普通 host | 锁页 host | GPU 稀疏 ID | 普通 host loader 最慢 rank，ms/步 |',
          '| --- | --- | --- | --- | ---: |']
for d in datasets:
    modes = [' / '.join(f'{v*1000:.2f}' for v in inputs[d][m]['step_seconds']) for m in ('pageable', 'pinned', 'gpu')]
    lines.append('| ' + ' | '.join([d] + modes + [f"{max(inputs[d]['pageable']['loader_ms_by_rank']):.3f}"]) + ' |')
lines += ['\n## 输入、启动与结果核对\n']
for d in datasets:
    r = results[d]
    for mode, data in inputs[d].items():
        if any(rank['read_bytes'] != 0 for run in data['io_by_run_rank'] for rank in run):
            raise ValueError(f'{d}/het_{mode}: physical reads observed in measurement window')
    for name, data in r.items():
        if name.startswith('_'):
            continue
        io = data.get('process_io_delta') or [x for run in data['io_by_run_rank'] for x in run]
        if any(x['read_bytes'] != 0 for x in io):
            raise ValueError(f'{d}/{name}: physical reads observed in measurement window')
    p = r['torch_prefetch']['mean_samples_per_second']
    g = r['torch_gpu']['mean_samples_per_second']
    h = inputs[d]['pageable']['mean_samples_per_second']
    hg = inputs[d]['gpu']['mean_samples_per_second']
    lines.append(f'- {d}：TorchRec 预取与 GPU 预放吞吐差 {(g/p-1)*100:.2f}%；GMP 普通与 GPU 稀疏输入吞吐差 {(hg/h-1)*100:.2f}%。')
    lines.append('  主对照及 GMP 输入消融各进程/各 rank 测量窗口物理 read_bytes 增量均为零。')
    for key in ('torch_prefetch', 'het'):
        starts = r[key]['input_startup_by_run_rank']
        load = [v['input_load_seconds'] for run in starts for v in run]
        prep = [v['input_prepare_seconds'] for run in starts for v in run]
        lines.append(f'  {key} 各 rank 启动输入加载 {min(load):.2f}–{max(load):.2f} s，输入准备 {min(prep):.2f}–{max(prep):.2f} s。')
    for key in ('reduction', 'unique', 'native', 'acp_only', 'materialized', 'online'):
        if key in r:
            times = [t for t in r[key]['input_setup_seconds'] if t is not None]
            if times:
                lines.append(f"  {key} 输入 setup {min(times):.2f}–{max(times):.2f} s（含计划等初始化）。")
    for name, check in r['_loss_checks'].items():
        lines.append(f"  {name}：{check['count']} 个 loss，最大绝对差 {check['max_abs']:.3g}。")
report = Path('/home/longjiu.bty/Rapid/内存驻留输入公平性与ACP实验报告.md')
with report.open('a') as f:
    f.write('\n'.join(lines) + '\n')
