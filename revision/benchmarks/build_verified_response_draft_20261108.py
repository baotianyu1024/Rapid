"""Create a conservative response draft from the audited original, without overwriting it."""
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / '中文Response.md'
target = ROOT / '中文Response_核验修订稿_20261108.md'
original = source.read_text()
text = original


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError(f'Expected one occurrence: {old[:70]}')
    text = text.replace(old, new, 1)


replace('尚待单机 8 卡训练验证的结果保留为空。',
        '当前已完成的 GPU 结果主要为子集重复短训；完整训练、物理扩展与历史结果证据的剩余缺口分别注明。')
replace('我们已运行可在本地执行的规划和通信统计；端到端吞吐量、延迟、显存及完整训练时间由单机 8 卡实验补充。',
        '我们已执行规划、通信统计和单机 8 卡短训；当前训练窗口吞吐不能替代含预处理的完整训练时间。')
replace('### 3）端到端比较与统一测量', '### 3）训练窗口比较与统一测量')
replace('TorchRec 强制全部 embedding 表使用 Row-wise 分片。模型适配后仍需检查计算一致性，',
        'TorchRec 强制全部 embedding 表使用 Row-wise 分片，这是受限策略对照，不能代称默认 AutoShard。当前复测硬件为 8×32 GiB V100-SXM2；设置 NCCL_P2P_DISABLE=1、NCCL_IB_DISABLE=1，与原论文 8×16 GB PCIe 配置分别报告。模型适配后仍需检查计算一致性，')
for name in ['T180', 'T200', 'T510']:
    old = f'| {name}（完整官方配置） | 显存不足 | 显存不足 | 显存不足 | 未完成全词表 | 无同配置 GPU 适配 | 无同配置 GPU 适配 | 显存不足 |'
    new = f'| {name}（完整词表；维度口径另述） | 未完成端到端验证 | 未完成端到端验证 | 未完成端到端验证 | 未完成全词表 | 无同配置 GPU 适配 | 无同配置 GPU 适配 | 未完成端到端验证 |'
    replace(old, new)
replace('T 系列完整官方词表的 AdaGrad 权重、梯度和状态均分下界超过每卡 32 GiB；',
        'T 系列须区分官方逐表维度与统一 dim32。三份完整权重大小数组并非所有实现的显存下界：八卡 T200 完整词表、dim32、逐元素 fused AdaGrad 的容量与更新探针已通过，每卡峰值约 24.78 GiB；它不包含完整网络和通信，不填为端到端结果。')
replace('表 R2：一次跨端传输只计一次，',
        '表 R2：真实数据三行保留为历史记录，尚未找回完全匹配的原始输入与完整证据链，不据此宣称本轮逐值复现；T 系列 RAPID 三项已从最终 binary 独立重数到显示精度一致。新输入下的真实数据 RAPID 重数与差异见[独立复核报告](中文Response独立复核_20261108.md)。一次跨端传输只计一次，')
replace('按官方完整词表，仅 FP32 embedding 权重 T180/T200/T510 分别需要约 247/387/427 GiB，本机 8×32 GiB V100 无法完成官方完整配置的端到端训练。',
        '按官方完整词表及官方逐表维度，仅 FP32 embedding 权重 T180/T200/T510 分别需要约 247/387/427 GiB；这不是统一 dim32 的权重大小。官方维度的 T200/T510 权重本身超过本机容量；T180 还需考虑 optimizer state 与运行缓冲。统一 dim32 下，权重加逐元素 AdaGrad 状态的理想每卡容量约为 33.706/24.778/26.676 GiB，不能由必须常驻全词表梯度的假设推断所有实现 OOM。')
replace('按表中显示值，RAPID 相对 HET-GMP 在 Terabyte、Kaggle 的通信分别减少约 4.79%、11.79%；Taobao 上则高出约 16.80%。我们保留收益与不利结果，并据此讨论适用范围。',
        '历史表显示值对应 RAPID 相对 HET-GMP 在 Terabyte、Kaggle 减少约 4.79%、11.79%，在 Taobao 增加约 16.80%；算术无误，但在原输入证据补齐前，不把这些幅度用作本轮已验证的比较结论。')
start = text.index('## Meta Revision 3：')
end = text.index('## Meta Revision 4：', start)
text = text[:start] + '''## Meta Revision 3：初始离线数据与增量更新的验证范围

感谢审稿人关于离线计划有效期的建议。当前可追溯的增量复测从 2,097,152 条初始样本开始，后续加入 56 个互不重叠、各 65,536 条的窗口。其末八窗口平均通信为 249,303.9375 向量／rank／8,192 条新样本，运行记录显示旧 embedding 的 owner 和 physical ID 没有变化。

该记录仅验证增量路径；历史四策略对照和稳定性序列尚未补齐，因此暂不报告原表 R3 的 120.32%、29.88% 及 CV、相对趋势等精确结论。这些窗口来自同日 shuffle 数据，检验的是数据增长，不是时间漂移。真实时间有序的 profiling/execution 窗口、同输入的无更新与全量重算对照，以及规划、映射分发和重编码耗时仍待完成。旧权重不迁移不等于更新总成本为零。证据范围见[独立复核报告](中文Response独立复核_20261108.md)。

''' + text[end:]
replace('我们在同一份 262,144 条 Criteo 输入上重新执行 RAPID，并在每个 local batch 下重新计算纯 Unique。',
        '以下为历史 262,144 条 Criteo 输入的 RAPID 与各 batch 纯 Unique 记录；本轮仅核对表内算术，尚未按原输入逐项重新运行。')
replace('表 R5：1,048,576 条 Criteo、local batch=8,192、5 epoch，单位为向量数。',
        '表 R5：历史 1,048,576 条 Criteo、local batch=8,192、5 epoch 的记录，单位为向量数；本轮未取得匹配原输入的逐值复现。当前表中只有 RAPID，以下归一化定义保留原方法说明，不据此声称已展示 HET-GMP 对照。')
replace('表 R6：权重统一按 dim=128、FP32 换算，HET-GMP 包含镜像；',
        '表 R6：历史放置记录，权重统一按 dim=128、FP32 换算，本轮已核对显示值的字节换算，但未重新运行原放置；当前表无 HET-GMP 行，不作含镜像对照。')
replace('我们在 262,144 条 Criteo、8 rank 上重新执行完整 RAPID，逐项改变参数，其余保持默认：',
        '以下为历史 262,144 条 Criteo、8 rank 完整 RAPID 参数记录；本轮未用原输入逐项重跑，保留为待复核数据。记录中逐项改变参数，其余保持默认：')
replace('| 设置 | 前反向向量／样本 | 本地命中率 | 主 emb 存储最大／平均 |',
        '| 设置 | 前反向向量／样本 | 去重 ID 请求本地命中率 | 主 emb 存储最大／平均 |')
start = text.index('感谢审稿人指出低支持次数可能影响共现统计。')
end = text.index('### 2）为什么在所述条件下保持模型计算', start)
text = text[:start] + '''感谢审稿人指出低支持次数可能影响共现统计。我们对 2,097,152 条 Criteo 源样本报告固定 multi-hot 工作负载的全目标结果，并新增保留每表第一列原始取模 ID、移除人工扩展的单 hot 对照。冷 ID 定义为出现频率低于 0.01%，对应支持次数不超过 209；同一 ID 在同一样本中重复出现只记一次。最低支持数只限制目标 ID，partner 仍是任意其他冷 ID。

$$
\\mathrm{MCR}(u)=\\max_{v\\in S_{\\mathrm{cold}},\\ v\\ne u}\\frac{\\mathrm{Count}(u\\cap v)}{\\mathrm{Count}(u)}.
$$

| 最小支持次数 | multi-hot 符合条件数 | multi-hot MCR>95% 数 | 比例 | 单 hot 符合条件数 | 单 hot MCR>95% 数 | 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1,886,103 | 1,333,054 | 70.6777% | 110,090 | 64,640 | 58.7156% |
| 20 | 953,848 | 770,892 | 80.8192% | 62,390 | 33,412 | 53.5535% |
| 50 | 353,257 | 299,515 | 84.7867% | 26,949 | 12,464 | 46.2503% |
| 100 | 129,222 | 110,490 | 85.5040% | 11,165 | 4,497 | 40.2777% |

表 R8：本轮可追溯输入的全量统计。multi-hot 读取现存全目标结果并核对计数；单 hot 新执行全目标枚举，另以独立 Python 对 16 个分层目标逐项复核，全部一致。两侧源样本相同，但 ID 总体不同，比例差不能解释为同一批 ID 的因果效应。固定扩展 lookup 会产生人工共现，因此我们将高比例结论限定于这一预处理工作负载，不直接推广到原始单 hot 数据。全目标枚举也不能消除有限窗口与选择最大 partner 的统计问题；独立时间窗口或零假设对照仍需补充。这些数与旧版 R8 的历史输入不同，不称为逐值复现；证据见[独立复核报告](中文Response独立复核_20261108.md)及其中的原始 JSON/二进制结果。

''' + text[end:]
replace('### 3）补充完整训练验证', '### 3）配对短训与完整训练的证据范围')
replace('LogLoss 顺序为参考实现 / RAPID。训练集仅',
        'LogLoss 顺序为参考实现 / RAPID，由 HugeCTR BCE eval 后的概率裁剪至 [1e-7, 1-1e-7] 后计算；不是未裁剪 logits 损失。训练集仅')
replace('### 1. Revision Request 1：充兼容基线与外部比较',
        '### 1. Revision Request 1：补充兼容基线与外部比较')
replace('我们将加入 TorchRec 自动分片与 EMBark 的实际训练比较，',
        '当前已有 TorchRec 全 Row-wise 与 EMBark 自动规划的实际训练窗口；TorchRec 默认 AutoShard 同配置对照仍需单列，')
replace('端到端结果由实际训练后填写，组件通信结果单独报告，',
        '已测训练窗口与尚缺的完整训练结果分别注明，组件通信结果单独报告，')
replace('请参见 Meta Revision 3。初始数据增至 2,097,152 条，随后加入 56 个窗口。增量只放置新 emb、调度新 sample，旧 emb 不迁移。末八窗口的增量通信为全量重算的 120.32%，相对纯 Unique 减少 73.96%。该结论限于本次同源数据增长；真实时间漂移与物理更新耗时仍需另行验证。',
        '请参见 Meta Revision 3。可追溯复测覆盖 2,097,152 条初始样本和 56 个新增窗口，记录旧 owner 与 physical ID 不变；四策略逐窗口证据未补齐，不报告历史比较百分比与稳定性结论。同日 shuffle 窗口不能替代时间漂移实验，物理更新、映射分发及重编码耗时仍需验证。')
replace('随后通过多数据集、多模型、五组配对 seed 的完整训练验证 AUC 和 LogLoss。',
        '目前提供多数据集、多模型、五组配对 seed 的 510 步短训 AUC 和 LogLoss；完整训练与逐步梯度/optimizer state 配对尚未完成。精度保持主张限定于推导列出的执行条件，区间跨零不作为等价证明。')
replace('对应 MCR>95% 的比例为 70.5613%、80.7066%、84.7580%、85.7500%。',
        '本轮 multi-hot 的 MCR>95% 比例为 70.6777%、80.8192%、84.7867%、85.5040%；同源单 hot 对照为 58.7156%、53.5535%、46.2503%、40.2777%，不将人工扩展下的高比例推广为原始数据的普遍性质。')
replace('RAPID 在两个 Criteo 子集上减少了通信，主 emb 分配也接近等量；这些是当前可观察到的优势。Taobao 上 GMP 仍更低，64 rank 的通信峰值也未全面优于 GMP，均会保留。',
        '历史记录显示两个 Criteo 子集存在通信收益、Taobao 存在不利结果；但在历史输入证据补齐前，不将这些幅度作本轮已验证结论。当前峰值表没有 HET-GMP 对照行，故不据此作 64 rank 峰值优劣判断。')

target.write_text(text)
diff = ''.join(difflib.unified_diff(original.splitlines(keepends=True), text.splitlines(keepends=True),
                                 fromfile=source.name, tofile=target.name))
(ROOT / 'response_audit_20261108_recheck/response_draft.diff').write_text(diff)
print(target)
