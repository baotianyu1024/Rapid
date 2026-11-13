"""Audit completed GPU runs against the two response tables."""

import csv
import re
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
RESPONSE = ROOT / '中文Response.md'
REPORT = ROOT / 'R1_R5最终审计_20261105.md'
SUPPLEMENT = ROOT / '两张表补测说明_20261104.md'
DATASETS = {'kaggle': 'Kaggle', 'terabyte': 'Terabyte', 'taobao': 'Taobao'}
MODELS = {'dlrm': 'DLRM', 'dcnv2': 'DLRM-DCNv2',
          'wide_deep': 'Wide&Deep', 'deepfm': 'DeepFM'}


def load_csv(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    document = RESPONSE.read_text()
    lines = document.splitlines()
    bagpipe = load_csv('bagpipe_d32_e2e_results_20261105.csv')
    herald = load_csv('herald_d32_e2e_results_20261105.csv')
    by_label = {}
    for line in lines:
        if line.startswith('| ') and len(line.split('|')) == 10:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            by_label[cells[0]] = cells
    issues = []
    for framework, rows, column in [('BagPipe', bagpipe, 6),
                                    ('Herald', herald, 5)]:
        for row in rows:
            label = f"{DATASETS[row['dataset']]}／{MODELS[row['model']]}"
            value = f"{float(row['samples_per_second']):,.0f}"
            if label not in by_label or not by_label[label][column].startswith(value):
                issues.append(f'{framework} {label}: CSV 与 R1 不一致')
    expected = {(dataset, model) for dataset in DATASETS for model in MODELS}
    measured_herald = {(row['dataset'], row['model']) for row in herald}
    measured_bagpipe = {(row['dataset'], row['model']) for row in bagpipe}
    missing_herald = sorted(expected - measured_herald)
    missing_bagpipe = sorted(expected - measured_bagpipe)
    r5_section = document.split('## Meta Revision 5：', 1)[1].split('## Meta Revision 6：', 1)[0]
    r5_rows = [line for line in r5_section.splitlines() if line.startswith('| ') and
               len(line.split('|')) == 7 and any(
                   line.startswith(f'| {dataset}／')
                   for dataset in DATASETS.values())]
    if len(r5_rows) != 12 or any('待填' in line for line in r5_rows):
        issues.append(f'R5 预期 12 行有效数据，实际 {len(r5_rows)} 行')
    stable_file = ROOT / 'het_kaggle_deepfm_stable_bce_20261105.csv'
    stable_het = stable_file.exists() and bool(load_csv(stable_file.name))
    if stable_het:
        stable_row = load_csv(stable_file.name)[0]
        expected_value = f"{float(stable_row['samples_per_second']):,.0f}"
        if not by_label['Kaggle／DeepFM'][4].startswith(expected_value):
            issues.append('HET-GMP Kaggle/DeepFM: CSV 与 R1 不一致')
    complete = (not missing_herald and not missing_bagpipe and
                stable_het and not issues)
    report = [
        '# R1/R5 最终审计',
        '',
        f'- 状态：{"真实数据短训表覆盖完整；仍有验证缺口" if complete else "仍有未完成项"}',
        f'- BagPipe 真实数据：{len(bagpipe)}/12 组 8 卡、510 步。',
        f'- Herald 真实数据：{len(herald)}/12 组 8 卡、510 步。',
        f'- HET-GMP Kaggle/DeepFM 稳定 BCE 的 SGD 0.01：{"完成" if stable_het else "未完成"}。',
        f'- R5 AUC/LogLoss：{len(r5_rows)}/12 行五种子短训数据。',
        '- T 系列须区分官方逐表维度与统一 dim32；全词表梯度数组不是 fused sparse AdaGrad 的必要容量。'
        'T200 dim32 的八卡完整词表容量更新探针已在每卡约24.78 GiB通过，但不代表端到端训练完成。'
        '详见 中文Response独立复核_20261108.md。',
        '',
        '## 审计范围外的未完成实验',
        '',
        '- R5 仅是重复使用 983,040 行的 510 步短训；全量数据最终收敛 AUC、LogLoss 和达到同等 AUC 的耗时尚未测完。',
        '- R1 并非所有框架都有三次独立、同优化器的可比测量；HET-GMP Kaggle/DeepFM 和 BagPipe 的部分 DeepFM 组合仍有高 loss，不能作为收敛精度排名。',
        '- T180/T200/T510 完整官方配置未获得端到端吞吐数字。',
        '',
        '## 表格校验问题',
        '',
    ]
    details = ([f'Herald {dataset}/{model}' for dataset, model in missing_herald] +
               [f'BagPipe {dataset}/{model}' for dataset, model in missing_bagpipe] +
               issues)
    if not stable_het:
        details.append('HET-GMP Kaggle/DeepFM 稳定 BCE 复测')
    report.extend(f'- {detail}' for detail in details)
    if not details:
        report.append('- 无。')
    report.extend(['', '逐项数值与测量限制见 [中文Response.md](中文Response.md)；原始汇总见 '
                   '[BagPipe CSV](bagpipe_d32_e2e_results_20261105.csv)、'
                   '[Herald CSV](herald_d32_e2e_results_20261105.csv) 和 '
                   '[R5 CSV](r5_paired_d32_summary_20261104.csv)。', ''])
    REPORT.write_text('\n'.join(report))
    supplement = SUPPLEMENT.read_text()
    supplement = re.sub(
        r'R1 的 105 个格子中，真实数据部分已有 \d+ 个完整训练窗口 GPU 吞吐数字',
        f'R1 的 105 个格子中，真实数据部分已有 '
        f'{71 + len(herald) + int(stable_het)} 个完整训练窗口 GPU 吞吐数字',
        supplement, count=1)
    SUPPLEMENT.write_text(supplement)
    print(f'R1_R5_AUDIT complete={complete} herald={len(herald)} '
          f'bagpipe={len(bagpipe)} r5={len(r5_rows)} stable_het={stable_het} '
          f'issues={len(issues)}')


if __name__ == '__main__':
    main()
