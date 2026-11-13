"""Fill the Response R1 matrix with measured 510-step runs and explicit gaps."""
import csv
from pathlib import Path


ROOT = Path('/home/longjiu.bty/Rapid')
RESPONSE = ROOT / '中文Response.md'
HEADER = '| 数据集／模型 | HugeCTR | TorchRec | EMBark | HET-GMP | Herald | BagPipe | RAPID |'
DATASETS = (('kaggle', 'Kaggle'), ('terabyte', 'Terabyte'), ('taobao', 'Taobao'))
MODELS = (('dlrm', 'DLRM'), ('dcnv2', 'DLRM-DCNv2'),
          ('wide_deep', 'Wide&Deep'), ('deepfm', 'DeepFM'))


def main():
    measured = {}
    for name in ('rowwise_d32_e2e_results_20261104.csv',
                 'het_d32_holdout_results_20261104.csv',
                 'het_d32_multimodel_results_20261104.csv'):
        for row in csv.DictReader((ROOT / name).open()):
            if row['dataset'] not in dict(DATASETS):
                continue
            method = row.get('framework') or 'HET-GMP'
            measured[row['dataset'], row['model'], method] = float(row['samples_per_second'])
    embark_csv = ROOT / 'embark_d32_e2e_results_20261104.csv'
    if embark_csv.exists():
        for row in csv.DictReader(embark_csv.open()):
            measured[row['dataset'], row['model'], 'EMBark'] = float(row['samples_per_second'])
    lines = [HEADER, '| --- | ---: | ---: | --- | ---: | --- | --- | ---: |']
    for dataset, label in DATASETS:
        for model, model_label in MODELS:
            values = []
            for framework in ('HugeCTR Unique', 'TorchRec Row-wise', 'HET-GMP', 'RAPID'):
                value = measured.get((dataset, model, framework))
                if value is None and framework != 'HET-GMP':
                    raise ValueError(f'missing measured {dataset}/{model}/{framework}')
                values.append(f'{value:,.0f}' if value is not None else 'NaN（SGD 0.01）')
            huge, torch, het, rapid = values
            embark = measured.get((dataset, model, 'EMBark'))
            embark_label = f'{embark:,.0f}' if embark is not None else '官方自动规划 GPU 适配中'
            lines.append(f'| {label}／{model_label} | {huge} | {torch} | {embark_label} | '
                         f'{het} | 无同配置 GPU 适配 | 无同配置 GPU 适配 | {rapid} |')
    for dataset in ('T180', 'T200', 'T510'):
        lines.append(f'| {dataset}（完整官方配置） | 显存不足 | 显存不足 | 显存不足 | '
                     '未完成全词表 | 无同配置 GPU 适配 | 无同配置 GPU 适配 | 显存不足 |')
    lines += [
        '',
        '表 R1：上表数字为当前 dim32、每卡 batch 8192、510 步的**单次 GPU 训练窗口吞吐**（samples/s）；EMBark 使用上游 HugeCTR EBC 自动规划器及其分片/压缩计划。HET-GMP 的共享 embedding 内核使用 SGD 0.01，其余已测框架使用 AdaGrad 0.01，因此不能直接作公平排名。Kaggle DeepFM 的 HET-GMP 在 SGD 0.01 下第 300 步起出现 NaN；SGD 0.001 的稳定性诊断另列。T 系列完整官方词表的 AdaGrad 权重、梯度和状态均分下界超过每卡 32 GiB；单分片活跃工作集吞吐见[扩展实验报告](扩展基线与T系列实测_20261104.md)。正式三次独立完整训练、同初始化与全量精度配对仍未完成；不可把本表称为正式性能排名。原始逐次日志见[主 CSV](rowwise_d32_e2e_results_20261104.csv)、[HET DCNv2 CSV](het_d32_holdout_results_20261104.csv)、[HET 多模型 CSV](het_d32_multimodel_results_20261104.csv)和[EMBark CSV](embark_d32_e2e_results_20261104.csv)；TorchRec 的 12 组三次重复及标准差见[复测 CSV](torchrec_rowwise_d32_repeats_20261104.csv)。',
        '',
        '',
        'Herald、BagPipe 的 GPU 适配排查，HET-GMP Kaggle/DeepFM 的稳定学习率诊断，以及 T 系列完整配置的不可运行原因，见[两张表补测说明](两张表补测说明_20261104.md)。',
        '',
    ]
    text = RESPONSE.read_text()
    start = text.index(HEADER)
    end = text.index('已完成的真实 8 卡 GPU 子集实测', start)
    RESPONSE.write_text(text[:start] + '\n'.join(lines) + text[end:])
    print(f'Updated {RESPONSE} with {12*4-1 + sum(k[2] == "EMBark" for k in measured)} measured real-data cells')


if __name__ == '__main__':
    main()
