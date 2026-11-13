# Response 修订范围判断

依据：`review.html` 中 Reviewer 2、3、5 的意见，以及投稿论文 `paper_vldb (2).pdf` 的摘要、Sections 3.2、4.1、5.1–5.2、6.1、7.1–7.8、8、9。这里区分三件事：论文中已有充分说明的内容只需在 response 指出；新增证据或原文错误要修改论文；当前尚缺实测的要求不能靠改写文字声称完成。

| 审稿意见与论文现状 | 处理 | 理由与准确位置 |
| --- | --- | --- |
| Reviewer 2 R1、Reviewer 3 W3、Reviewer 5 D1：外部系统对照不足；论文 Section 7.1 只以 HugeCTR 为主要基线，Section 8 没有讨论最接近的 HET-GMP。 | **改论文** | 新增系统结果应进入 Section 7.1–7.2；HET-GMP 的共现图放置与 RAPID 的区别应进入 Section 8。response 同时列完整数据，方便审稿人直接核对。 |
| Row-wise TorchRec、HET-GMP 使用 SGD 等配置问题。 | **合并到一次论文修改** | 这些是新增基线的复现条件，只需在 Section 7.1 或新表注交代一次；response 开头概述即可，无须为每项单独贴一段“论文修改”。 |
| Reviewer 2 R1、Reviewer 5 D2：希望更多负载；论文原本已经有 Taobao，不能称其为新增数据域。T 系列目前仅有缩容分片的协议通信计数。 | **数据若纳入论文，则改评估；OOM 原因在 response 回答** | T 分片生成方式、dim32/batch512 与通信表应一同放入评估章节，否则正文无法支持新增的 T 系列结论。为何官方完整配置未做端到端，可在 response 说明并在表注限定。不能把 T 分片写成新增物理吞吐或新真实数据域。 |
| Reviewer 2 R2、Reviewer 3 D1：16–128 GPU 缺少物理扩展实验。论文 Section 7.6 已明确“no measured end-to-end runtime beyond the 8-GPU physical cluster”。 | **只改摘要与 Section 9；Section 7.6 直接回答** | 审稿人 2 明说 Section 7.6 已足够明确，要求将边界一致地写入摘要和结论。重复替换 Section 7.6 没有必要。物理多机吞吐仍未测得，不能声称已补足。 |
| Reviewer 2 R3：较早 profiling 与较晚执行窗口、增量更新、迁移成本。论文 Section 7.8 只说支持增量且“minimal additional overhead”，未给实测。 | **改论文，同时明确证据边界** | 应在 Section 7.8 加入同输入四策略结果、旧行零迁移，并改写无实测支撑的“minimal overhead”。现有打乱的同日窗口只验证数据追加和通信，不是实际时间漂移，也未计规划/分发时间；这些缺项不能用文字代替。 |
| 480,344.75 与约 960,689 的差异。 | **response 直接澄清** | 前者是每 rank 每步**单向前向远端接收**，后者是相应前向加反向合计；只需在表题或表注保持口径，不必在论文额外写一整段解释。 |
| Reviewer 2 R4：batch size、最忙 rank 通信与存储。论文只有单档 batch 和“峰值高 10%–20%”的概括。 | **改论文** | 在评估章节加入 batch 通信表及最忙 rank 的绝对量、主 embedding 行数和峰均比；response 可直接用同一表回答。 |
| Reviewer 2 D4 还要求各 batch 的训练吞吐、local-hit rate、可能的梯度累积。当前只有通信计数。 | **证据未齐，不能写成已完成** | 新增训练或回放实测后才能报告这些指标。现阶段在 response 准确说明表只检验传输量；不能由更低传输量推断逐档吞吐。 |
| Reviewer 3 W4：β、γ、Top-K、METIS seed 比例和顺序。论文 Section 5.2 给 seed 比例举例，但没完整列默认值和敏感性。 | **部分改论文、部分仍需证据** | Section 5.2 应写实际默认参数；评估章节加入已测的 β、γ、Top-K 与 profiling 顺序结果。当前没有 METIS seed 比例的消融或质量界，不能声称该项已回答。 |
| Reviewer 3 D2：冷 ID MCR 是否由极少支持次数夸大。论文 Section 5.1.2 对约一百万样本给出 79.3%，并将共现描述得较普遍。 | **改论文，同时收紧主张** | 增加最低支持次数实验及定义；把结论限定为所用 Criteo multi-hot 负载。因为生成的 multi-hot 扩展可能直接带来共现，仅过滤低支持次数仍不足以证明自然数据或时间稳定性。 |
| Reviewer 2 R5：一日单次 AUC 不足。论文 Section 4.1 的语义论证较绝对，Section 7.3 只给一次一日测试。 | **改论文** | Section 4.1 写明 global batch 不变的适用条件；Section 7.3 增加完整 Taobao 一轮三种子结果，并明确 12 组是 510 步短训。response 中可解释配对区间；“区间含零”不能写成等效证明。 |
| Reviewer 5 D3：Equation (5) 与 Figure 4(b)/(c)、Figure 11 的 Reduction 口径不一致。 | **必须改论文图和公式说明** | Figure 4(b)–(c) 与 Figure 11 的全 Reduction 曲线应按远端口径修正，Section 6.1 在公式 (5) 后交代 BG 包含本地 B、实际远端是 B(G−1)。response 给出 57,344/表和 1,490,944/26 表即可；HugeCTR 源码链接是回复中的证据，不必写入论文正文。 |
| HET-GMP 比较中的不同优化器、框架开销，以及 AUC 区间的解释。 | **response 直接回答，论文只需必要表注** | 系统比较的优化器设置要公开；具体差距为何不能全归因于通信，可在审稿回复中回答。置信区间的统计含义也只需按问题解释，无须为此增加整段方法论文文字。 |

## 当前 response 的修改原则

1. 保留原 `中文Response.md` 的六项 Meta Revision、主要数据表和逐审稿人结构。
2. 只对外部基线、增量更新、batch/负载/参数、冷 ID/精度、Figure 4 修正，以及摘要/结论的扩展边界，附有必要的论文文字或图表修改位置。
3. Section 7.6 的边界、T 系列未跑完整配置的原因、通信单向/双向口径和源码证据，在 response 中直接回答；不再为这些说明重复制造论文段落。
4. 对尚缺的多机物理吞吐、时间漂移、batch 吞吐/local-hit/梯度累积、METIS seed 消融，不写成“已经解决”。

当前英文和中文审阅稿已按上述原则去掉了单独的“通用实验设置”论文修改段落，以及对 Section 7.6 的重复替换；相应配置并入新增基线的 Section 7.1–7.2 修订文字。
