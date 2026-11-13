# HugeCTR 全 Reduction 与 RAPID：8 卡训练吞吐复测

本轮先保持现有模型 shape 不变，把两个多热 Criteo 工作负载的 HugeCTR 基线从全 Unique 改为论文采用的全 Reduction。两侧均使用 8 张 V100-SXM2 32 GiB、每卡 batch 8,192、embedding dim 32、FP32、AdaGrad 0.01、同一份预打乱的 983,040 条训练样本和 510 步；训练循环重复这 15 个 global batch。Kaggle 使用当前 DCNv2 实现；Terabyte 先测试当前较轻的 DLRM，再按论文附带的 `code/Rapid/training/run_hugectr_dlrm.py` 所用三层 Cross Network 和稠密 MLP shape 测 DCNv2。RAPID 使用已生成的 CPS/ACP 计划；Kaggle 25 张表 Unique、1 张 Reduction，Terabyte 26 张表 Unique。

吞吐以完整 510 步 `model.fit` 墙钟计算，包含训练时读取、前后向、embedding 通信与更新；不含离线计划、模型初始化和验证。各配置独立运行一次；这不是包含离线准备的作业总时间。训练窗口六次运行的物理 `read_bytes` 均为 0。命令见 `code/run_hugectr_reduction_vs_rapid_20261109.sh`，逐次日志在 `/tmp/rapid_reduction_vs_rapid_20261109/`。Terabyte/DCNv2 使用 `RUN_DATASETS=terabyte RUN_MODEL=dcnv2` 调用脚本。

| 工作负载 | HugeCTR 全 Reduction | RAPID | RAPID / HugeCTR | 论文 Figure 8 |
| --- | ---: | ---: | ---: | ---: |
| Kaggle / DCNv2 | 547,077 samples/s；61.094 s | 1,381,186 samples/s；24.199 s | 2.525x | DCN：601,000 / 1,840,000 samples/s；3.062x |
| Terabyte / DLRM | 635,920 samples/s；52.559 s | 2,010,674 samples/s；16.623 s | 3.162x | DLRM：569,000 / 1,386,000 samples/s；2.436x |
| Terabyte / DCNv2 | 546,769 samples/s；61.129 s | 1,329,655 samples/s；25.137 s | 2.432x | DLRM/DCNv2：569,000 / 1,386,000 samples/s；2.436x |

全 Reduction 使当前 Kaggle 的 HugeCTR 吞吐接近论文基线，但 RAPID 的提升为 2.525x，而非论文的 3.062x。Terabyte 使用较轻 DLRM 时得到 3.162x；改成论文附带实现的 DCNv2 风格网络后为 **2.432x**，与论文的 **2.436x** 接近，双方绝对吞吐均比论文低约 4%。这支持 Terabyte 的原差异主要来自模型计算量和基线策略，Kaggle 的 RAPID 差距仍需继续定位。

为定位模型计算量，另对六种配置各采 40 步 Nsight Systems trace。下表是八卡 GPU kernel 累计时间除以 8 卡和 40 步；NCCL kernel 与其他 kernel 分组。它用于诊断，并非严格的互斥墙钟分解，不能把 NCCL 百分比当成实测网络忙碌时间。无 profiler 的吞吐以上表为准。

| 工作负载 / 方法 | 其他 kernel ms/步 | NCCL kernel ms/步 | NCCL 占累计 kernel 时间 |
| --- | ---: | ---: | ---: |
| Kaggle / HugeCTR Reduction | 28.47 | 91.36 | 76.2% |
| Kaggle / RAPID | 29.38 | 17.58 | 37.4% |
| Terabyte / HugeCTR Reduction | 16.85 | 86.16 | 83.6% |
| Terabyte / RAPID | 19.39 | 12.06 | 38.3% |
| Terabyte DCNv2 / HugeCTR Reduction | 28.44 | 91.38 | 76.3% |
| Terabyte DCNv2 / RAPID | 31.00 | 17.44 | 36.0% |

论文 Figure 9 的 Kaggle 计算/传输为 HugeCTR 25.0/84.0 ms、RAPID 24.8/9.8 ms，传输占比分别为 77.1%/28.3%；当前 Kaggle 计算略高，RAPID 的 NCCL kernel 时间也更高。论文的 Terabyte 为 HugeCTR 32.1/83.0 ms、RAPID 32.3/15.0 ms，传输占比为 72.1%/31.7%；当前较轻 DLRM 的非 NCCL kernel 时间明显偏低。换成附带实现的 DCNv2 风格网络后，Terabyte 的其他/NCCL kernel 变为 HugeCTR 28.44/91.38 ms、RAPID 31.00/17.44 ms，已接近论文的时间量级。Kaggle RAPID 的通信时间仍高于论文；单独缩小模型会提高通信占比，不能消除这部分差距。论文正文没有完整列出逐层 shape，因此本轮把附带实现作为 Terabyte 网络依据，同时保留硬件、输入切片和 profiler 口径差异。

本轮没有改动已有 Response 表或数据集。原始 profiler 文件位于 `/tmp/rapid_shape_{reduction,rapid}_{kaggle,terabyte}*.nsys-rep`。
