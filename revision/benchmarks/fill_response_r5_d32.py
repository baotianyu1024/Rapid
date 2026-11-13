"""Insert audited five-seed short-train pairs into the Response R5 matrix."""
import csv
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
RESPONSE = ROOT / '中文Response.md'
SUMMARY = ROOT / 'r5_paired_d32_summary_20261104.csv'
HEADER = '| 数据集／模型 | 参考实现 AUC | RAPID AUC | AUC 配对差及区间 | 两者 LogLoss |'
DATASETS = (('kaggle', 'Kaggle'), ('terabyte', 'Terabyte'), ('taobao', 'Taobao'))
MODELS = (('dlrm', 'DLRM'), ('dcnv2', 'DLRM-DCNv2'),
          ('wide_deep', 'Wide&Deep'), ('deepfm', 'DeepFM'))


def main():
    measured = {(row['dataset'], row['model']): row
                for row in csv.DictReader(SUMMARY.open())}
    lines = [HEADER, '| --- | ---: | ---: | --- | --- |']
    for dataset, dataset_label in DATASETS:
        for model, model_label in MODELS:
            row = measured.get((dataset, model))
            if row is None:
                lines.append(f'| {dataset_label}／{model_label} | 实验进行中 | 实验进行中 | 实验进行中 | 实验进行中 |')
                continue
            diff = (f'{float(row["auc_diff"]):+.6f} '
                    f'[{float(row["auc_diff_ci95_low"]):+.6f}, '
                    f'{float(row["auc_diff_ci95_high"]):+.6f}]')
            lines.append(f'| {dataset_label}／{model_label} | '
                         f'{float(row["reference_auc"]):.6f} | '
                         f'{float(row["rapid_auc"]):.6f} | {diff} | '
                         f'{float(row["reference_logloss"]):.6f} / '
                         f'{float(row["rapid_logloss"]):.6f} |')
    caption = ('表 R5a：当前可复核的是每组 **5 个种子、510 步短训**的均值；'
               '参考实现为 HugeCTR 全 Unique，使用同一预打乱训练集、65,536 行留出集、'
               '零 embedding 初始化、AdaGrad 0.01、dim32 和每卡 batch 8,192。'
               '区间是 RAPID 减参考实现的双侧 95% 配对 t 区间；'
               'LogLoss 顺序为参考实现 / RAPID。训练集仅 983,040 行且反复读取，'
               '这些数字不能视为完整数据集最终收敛 AUC、精度等价性证明。'
               '逐次实测见 [R5 运行 CSV](r5_paired_d32_runs_20261104.csv) 和'
               '[配对汇总 CSV](r5_paired_d32_summary_20261104.csv)。')
    text = RESPONSE.read_text()
    start = text.index(HEADER)
    end = text.index('现有 Kaggle、Taobao 与 Terabyte 的短训练留出集 AUC', start)
    RESPONSE.write_text(text[:start] + '\n'.join(lines) + '\n\n' + caption + '\n\n' + text[end:])
    print(f'Filled {len(measured)} of 12 R5 short-train rows')


if __name__ == '__main__':
    main()
