# 实验与证据索引

目前最值得保留的结果是**状态与指令监督使文本解析和迭代执行学会了所定义的任务**。在模板化的单比特布尔规则链上，R3-State 比相同架构的仅答案监督组高 49.0 个百分点，同时获得正确的执行轨迹和反事实状态控制。该监督包包含初值、算子、gate 与下一状态四类标签；现有实验尚未分离它们各自的贡献。

后续同监督接口比较将二维连续、软反馈和硬反馈都训练到相近水平，Support-Hole 则显示直接反事实状态标签可以复现并超过当前 gold-state IIT 的收益。因此本仓库把状态与指令监督的完整训练流程作为主结果，把接口形式和 IIT 作为归因对照。

百分比均为准确率；`ID` 指训练深度范围内的独立测试集，`OOD` 指更长或留出组合的测试集。跨阶段的数字不能直接相减作为某个模块的净收益。小型结果文件保存在 [`results/evidence`](../results/evidence/)；旧阶段的完整脚本仍在原项目中，大型特征缓存和模型权重保留在原计算主机。

## 主结果：状态与指令监督

此阶段换用**新的规则单族数据和新模型**：5,000 训练、1,000 验证、100 长深度 stress、1,000 ID、1,000 OOD；训练/ID 深度 1–3，OOD 深度 4、5、6、8、10（每深度 200）。从同一 merged Kev-0.8B 起点训练最后六个文本块与共享 typed parser；一次完整输入前向预测初值、operator、gate。共享转移模块在每一步只能读当前二值状态和一条静态指令，答案头只能读末状态与可见选项。主比较仅种子 11，batch 8、4 epoch，验证答案准确率再按 NLL 选 checkpoint。`S` 是初值、operator、gate、转移 logits 的平均 CE；`I` 是有效非 donor-copy 反事实答案 CE。

| 模型与训练信号 | ID / OOD / OOD 深度 10 答案 | 机制证据与性质 |
| --- | --- | --- |
| Deep，`A+S`，十个独立绝对步块 | 100 / 49.6 / 49.5% | **新锁定测试，单种子。**深度 4–10 的块从未在深度 1–3 训练过，不能据此证明共享递推优于同样受训的非共享计算。[测试](../results/evidence/hard_iit/trial_deep_ood_test.json) |
| 共享 R3，仅答案 `A` | 50.1 / 50.4 / 50.5% | 64 题 smoke 最好仅 33/64；短题优化/状态获取已失败，不能否定有可读状态时的递推。[smoke](../results/evidence/hard_iit/trial_r3_smoke.json) · [测试](../results/evidence/hard_iit/trial_r3_ood_test.json) |
| 共享 R3-State，`A+S` | 100 / **99.4** / 97.0%；OOD 完整程序 97.7%、完整轨迹 99.1%；有效非 donor-copy 干预 584/589 正确、578/589 正向翻转 | **新锁定测试，单种子。**同结构的 answer-only 对照支持完整解析与执行监督包的作用；不是仅累积状态标签的消融。[选择](../results/evidence/hard_iit/trial_r3_state_selection.json) · [测试](../results/evidence/hard_iit/trial_r3_state_ood_test.json) |
| 共享 R3-IIT，`A+S+I` | 100 / **100** / 100%；有效干预 588/589 正确且翻转 | IIT 比 R3-State 多编码反事实例子，训练暴露/算力更高。+0.6 点全部是深度 10 的六题；固定 checkpoint 的 encoder/core 交叉组合显示改正跟随 **encoder**，不跟随 transition。R3-State 选 epoch 1，IIT 选 epoch 4，事后等 epoch 重训可得到 199/200 深度 10；故不支持 IIT 独有的转移改善。[测试](../results/evidence/hard_iit/trial_r3_iit_ood_test.json) · [交叉组合](../results/evidence/hard_iit/posthoc_cross_composition_d10.json) · [事后重训](../results/evidence/hard_iit/posthoc_equal_epoch_replay.json) |
| Gold-State R3，oracle typed 输入，执行器仅答案 CE | 100 / 100 / 100%；长深度 stress 答案与轨迹 100/100 | **独立正控制**，仅 770 个核心参数，无逐步状态损失；batch 256、lr 0.003、上限 60 epoch，以验证准确率选择并在达到 98% 后停止，第 3 epoch 入选。不能据主实验认定累积状态标签对执行器必需。[训练入口](../kev/latent_iit_run.py) · [选择](../results/evidence/hard_iit/gold_rule_selection.json) · [stress](../results/evidence/hard_iit/gold_rule_stress.json) · [测试](../results/evidence/hard_iit/trial_gold_ood_test.json) |

主实验的具体正结果是：初值和程序解析从近机会水平提升到完整程序 977/1,000，完整执行轨迹从 35/1,000 提升到 991/1,000，最终答案从 504/1,000 提升到 994/1,000。模型预测状态的干预测试又验证了执行语义。监督过程使预定义状态接口能够工作；接口本身以及最终只读终态的路径由设计规定。[仅答案指标](../results/evidence/hard_iit/trial_r3_ood_test.json) · [状态监督指标](../results/evidence/hard_iit/trial_r3_state_ood_test.json)

**任务结构的事后审计。**成功任务包含 copy、invert、and(gate)、or(gate)，动态状态为一位。训练规则题的 9,998 次局部转移覆盖全部 12 种合法输入，最少一类也有 603 次；OOD 的 6,600 次转移仍使用同一集合。这是已见局部转移的长度外推。and(false) 和 or(true) 会擦除此前状态，所以十条规则不一定构成十步持续依赖。

| OOD 深度 | 包含常量重置 | 不含重置 |
| --- | ---: | ---: |
| 4 | 145/200 | 55/200 |
| 5 | 153/200 | 47/200 |
| 6 | 161/200 | 39/200 |
| 8 | 185/200 | 15/200 |
| 10 | 187/200 | 13/200 |

深度 10 有 116/200 题可由最后重置起的至多三条规则确定答案。无重置子集的 R3-State 答案为 12/13；这是小样本的事后描述，历史逐题文件未保存可重分组的轨迹。审计只重建确定性数据和读取现有预测，五个划分的完整/规则 JSONL 共十个哈希均匹配原数据清单。没有重新训练。[审计定义及计数](../config/data_manifest.json) · [实际例题](METHOD.md)

这些结果给出了一个可运行、可逐步检查的监督接口基线；它尚未验证更大的状态空间、未见局部规则或真实文本中的自主程序发现。完整设置、世界隔离和冻结清单见[配置](../results/evidence/hard_iit/config.json)与[冻结清单](../results/evidence/hard_iit/frozen_manifest.json)。

## 同任务、同监督的状态接口归因

随后在**相同规则生成任务的新种子 20261001**上固定上一步单次读取 Kev parser，给三个 770 参数执行器缓存完全相同的预测初值、operator、gate；训练/验证/ID/OOD 仍为 5,000/1,000/1,000/1,000，另有 100 条 stress。A 将二维原始状态 logits 反馈到下一步，B 反馈 softmax 分布，C 反馈直通估计的硬 one-hot；三组答案头都只读终点硬类别。种子 11/23/37、AdamW、batch 128、lr 0.003、`答案 CE+0.25×逐步状态 CE`，无 teacher forcing、无 IIT。12 epoch 的开发版中 B 短深度未通过门槛；**开锁前仅做一次修复**，三组从同初始化统一改为 30 epoch。所选 checkpoint 全部在验证深度 1/2/3 达到 99.4/99.4/99.7%，随后才解锁新测试。

| 同监督执行器 | ID | OOD 全部 | OOD 深度 10 | 深度 6/8/10 | 证据来源 |
| --- | ---: | ---: | ---: | ---: | --- |
| A：二维连续 logits | 99.4% | 98.1% | 94.67% | 97.17% | [锁定结果](../results/evidence/state_interface/locked_results.json) |
| B：软状态反馈 | 99.4% | 98.2% | 95.0% | 97.33% | [锁定结果](../results/evidence/state_interface/locked_results.json) |
| C：硬状态反馈 | 99.4% | 98.2% | 95.0% | 97.33% | [锁定结果](../results/evidence/state_interface/locked_results.json) |

B、C 在三种子全部 OOD 题上的答案预测完全相同。晚深度 B−A 与 C−A 均为 **+0.167 点**，按 300 个独立世界聚类的 95% 区间 **[0,+0.389] 点**；C−B 为零。冻结 parser 的整条程序精确率 89.0%，但相同 parser 预测交给精确解释器后答案达 98.2%，B/C 逐题与该解释器一致。模型状态替换的有效非 donor-copy 事件中，B/C 为 593/626 oracle 正确、581/626 正向翻转；在 recipient 解析正确且 donor 状态正确的子集为 542/542；同状态但不同前缀的条件化空替换为 0/2,281 翻转。所有计数共享题目/世界，不能把事件当成独立样本。

预注册的 ≥5 点继续投入门槛失败；B/C 与 A 的区间都落在 ±2 点内。因此**该小型、接近满分的规则任务没有显示反馈或硬离散化的独立收益**。A 只有二维 logits，并不是宽维连续记忆；C 使用独立训练后冻结的 parser，不能把其 98.2% 当作前一轮联合训练 99.4% 的直接复现。按照既定迁移门槛，未运行关系/lookup 迁移。[设置](../results/evidence/state_interface/config_portable.json) · [冻结清单](../results/evidence/state_interface/frozen_manifest_portable.json)

## Support-Hole：IIT 的支持补全诊断

这是另一个**与 Kev/语言解析分离的**有限状态控制：16 个状态、8 条固定随机置换规则，共 128 个合法 `(state, rule)` pair；38 个 pair 从 factual 训练轨迹完全留出，但各状态与各规则仍各自出现。用 gold 初值、gold 指令与 gold donor 状态，只训练 1–3 步转移；测试长度 4–10 的留出组合路径。数据种子 20260929；2,753 条 factual 训练程序、689 条 seen-only 验证程序；五个配对训练种子 11/23/37/41/53。所有组先在 90 个 seen pair 上共同 warm start 125 step，再各训练 400 step。IIT 与 CF-Aug 接收完全相同的 1,216 个反事实 tuple；CF-Aug 直接监督相同 swapped pair 的下一状态，IIT 监督 recipient 后缀的终点状态。整轮 CPU 用时约 23 秒，无 GPU。

| 训练组 | 留出 pair 一步准确率，五种子平均 | 含 1–5 个 hole 的深度 10 答案 | 解释与来源 |
| --- | ---: | ---: | --- |
| State：仅 factual 状态转移 | 1.05% | 5.53% | 见过的组合仍为 90/90；随机置换没有给出可推断的规律，许多留出映射也无法由其余 pair 唯一确定。[结果](../results/evidence/support_hole/results.json) |
| State + IIT：gold 状态交换、后缀终点 CE | 95.26% | 86.62% | 新 pair 获得训练信息；仍有部分局部错误在长链累积。[结果](../results/evidence/support_hole/results.json) |
| State + CF-Aug：相同 tuple 的局部下一状态 CE | **100%** | **100%** | 同样的新 pair 信息，局部标签更直接；五种子均满分。[结果](../results/evidence/support_hole/results.json) |
| State + IIT-seen：仅已见 pair 的交换 | 1.05% | 5.48% | 增加交换样本但未补足 pair 支持。[结果](../results/evidence/support_hole/results.json) |

所有组的 seen-only 浅验证为 689/689、seen-only 深度 10 为 256/256。预定的“IIT 比 matched CF-Aug 在留出 pair 与深度 10 上均稳定高至少 5 点”门槛失败。**这里训练直接注入 oracle donor 状态，没有计算或训练 donor 模型激活**；相同 gold swap 的后缀答案损失本来就等价于普通带相同后缀标签的样本增强。四组还同时改变了监督标签的局部性，不能据此证明 full IIT 的一般效能或训练优化的固有排名。结果支持的窄结论是：此显式状态任务中的主要提升来自为原 factual 数据缺失的组合提供监督。[逐种子结果与留出列表](../results/evidence/support_hole/results.json) · [完整性 smoke](../results/evidence/support_hole/smoke.json)

## 历史实验：自由潜状态与 gold 输入

第一批实验使用同一批 16,000 道世界隔离的合成题：10,000 训练、2,000 验证、2,000 ID 测试、2,000 OOD 测试。关系链是四选一，布尔规则链是二选一，各占一半；混合随机基准为 37.5%。训练题约 90% 为深度 1–3，10% 为深度 4–6；OOD 为深度 4–6，包含训练中留出的相邻关系或规则组合。数据种子为 20260929。**Read-Once、Gold-H、Gold-Z1 和后续状态诊断反复使用了已打开的 OOD 集，均为探索性结果。**

| 实验与模型 | 输入、监督与训练设置 | 主要结果 | 证据性质与来源 |
| --- | --- | --- | --- |
| Frozen-Kev scratchpad：Original、FinalOnly、Process、Deep | 冻结 `jaredpalmer/kev-0.8b` 主干和 LoRA。Original 是在本合成任务上**重新训练的原指针头**，不是未经训练的发布模型。其余模型在四个 256 维 slot 上运行四次；FinalOnly 仅答案 CE，Process 与 Deep 还预测中间状态。种子 11/23/37；AdamW，3 epoch，batch 64，lr 5e-4；辅助权重从验证集选择。 | ID 按表中顺序为 76.53/78.00/77.42/75.92%；OOD 为 41.10/41.03/40.63/40.97%。Process−FinalOnly −0.40 个百分点，按世界配对的 95% 区间 [−1.37,+0.62]。Process 的有效整状态交换仅 0.30% 使最高分选项转向反事实答案。 | **原始锁定测试，3 种子；负结果。**上下文化 option/decide 表征仍可直接携带证据，交换 `M_t` 不能隔离全部答案路径。[配置](../results/evidence/latent_scratchpad/config.json) · [汇总](../results/evidence/latent_scratchpad/aggregate.json) · [配对区间](../results/evidence/latent_scratchpad/paired_intervals.json) · [干预](../results/evidence/latent_scratchpad/intervention_results.json) |
| Read-Once：Bottleneck-1、R3、DeepHead | 冻结 Kev 的状态前缀 `H`；问题和选项单独编码。四个 1024 维 slot 只读 `H` 一次，随后不再接触原始输入；R3 共享两次后续 Transformer，Deep 用独立 MLP。仅答案 CE；同样 3 种子、3 epoch、batch 64、lr 5e-4。 | ID 37.47/37.40/37.50%，OOD 37.02/37.57/37.50%；R3−B1 +0.55 点，区间 [−0.02,+1.07]。R3 的 ID 深度 1 仅 37.23%。fp32 的 OOD 相关与无关状态编辑均为 **0/576** 次预测改变。 | **重用 OOD 的探索性测试；负结果。**已经移除旧答案旁路，但短题也未学会，因此不能把失败归于循环。干预特征经一致的 state-only 提取路径，与主测试缓存路径有数值差异。[配置](../results/evidence/read_once/config.json) · [汇总](../results/evidence/read_once/aggregate.json) · [区间](../results/evidence/read_once/paired_intervals.json) · [fp32 干预](../results/evidence/read_once/transplant_results_fp32.json) |
| Gold-H：B1、R3、Deep | 用确定性的 gold 原子事实、问题与选项替换 Kev 表征，仍须经过原模型的一次学习式读取形成 `Z1`；只用答案 CE。3 种子，最多 15 epoch，batch 64，lr 5e-4。 | ID 37.35/37.33/38.10%，OOD 37.20/37.08/36.88%。64 题训练集的 B1/R3 smoke 均可过拟合至 100%。 | **探索性 gold 输入对照。**完整原子事实可以被符号求解器使用，但这里的学习式读取与后续计算仍混在一起；full-data 失败不能单独否定下游递推。[配置](../results/evidence/gold_state/config.json) · [汇总](../results/evidence/gold_state/aggregate.json) · [区间](../results/evidence/gold_state/paired_intervals.json) · [B1 smoke](../results/evidence/gold_state/smoke_bottleneck1_seed11.json) · [R3 smoke](../results/evidence/gold_state/smoke_recurrent3_seed11.json) |
| Gold-Z1：B1、R3、Deep | **仅布尔规则题**。直接把被问主体的初值、每一步 operator/gate 以及每步 `初值×operator×gate` 联合坐标放入选项条件化的 `Z1`，绕过读取模块；没有中间累积值、最终真值或答案标签。沿用原 `resume_from`，仅答案 CE；3 种子，最多 15 epoch，batch 64，lr 5e-4。 | 64 题训练 smoke 和三组验证深度 1 均达到 100%。规则 ID 91.07/99.87/99.90%，规则 OOD 60.27/67.97/72.43%。R3−B1 +7.70 点，区间 [+4.30,+10.87]；R3−Deep −4.47 点，区间 [−7.07,−1.83]。R3 的相关/同答案无关整 `Z1` 替换在 ID 上改变预测 99.7%/0.1%，OOD 上为 53.7%/41.9%。 | **重用 OOD 的规则计算对照。**联合坐标让一步真值表更易线性读取。+7.70 点是**同样 gold 输入下 R3 对 B1**，并非 gold 注入本身的收益；Deep 总体更强。整个 `Z1` 替换等同直接使用 donor，不能证明中间状态递推。[配置](../results/evidence/gold_state_z1/config.json) · [选择及逐深度验证](../results/evidence/gold_state_z1/selection.json) · [R3 smoke](../results/evidence/gold_state_z1/smoke_recurrent3_seed11.json) · [汇总](../results/evidence/gold_state_z1/aggregate.json) · [区间](../results/evidence/gold_state_z1/paired_intervals.json) · [替换](../results/evidence/gold_state_z1/transplant.json) |

### 冻结表征与辅助状态监督诊断

| 诊断 | 设置与实际结果 | 解释与来源 |
| --- | --- | --- |
| 冻结 Kev `H+q` probe | 一种浅注意力读取加两层 MLP；输入仅状态前缀 `H` 与独立问题 `q`，不用 contextual decide、选项、trace 或答案。种子 11，15 epoch，batch 64，lr 1e-3，以验证损失选 checkpoint。ID 规则初值 **82.9%**（q-only 54.3%），第一 gate **61.8%**（q-only 49.9%）；关系第一跳 `x1` **2.10%**（类型条件均匀参考约 2.60%）。 | 规则中的某些原子变量可恢复；不能推断所有计算状态已可用。15 epoch probe 与早期 6 epoch 未收敛 screen 要分开。[主 probe](../results/evidence/state_diagnosis/probe_primary15/probe_metrics.json) · [控制](../results/evidence/state_diagnosis/probe_controls15/probe_metrics.json) |
| 关系第一跳 key/value 正控制 | 同一个约 199,680 参数读出器和初始化，分别读 gold 原子边与冻结 Kev `H`。单种子 15 epoch 上限；gold-H 在 ID/OOD 都为 **100%**，冻结 H 为 **2.40%/2.00%**，接近类型条件机会率。 | 初始化按 gold 边的 key/value 坐标设计，偏向 gold 表征。结果是该读出接口和预算下的**未恢复**，不是 Kev 中信息不存在的证明。[结果](../results/evidence/state_diagnosis/probe_keyvalue_seed11/probe_keyvalue_metrics.json) |
| 自由 `Z1` 的完整辅助标签 | 在原 B1/R3/Deep 的 `Z1` 上加状态解码头，损失 `L_answer+1.0L_state`，标签为所有非最终关系状态、规则初值/有序 operator/已观察 gate/非最终累积值；**解码结果不反馈至答案路径**。种子 11，6 epoch，batch 64，lr 5e-4；匹配的 λ=0 控制。R3 验证深度 4–6 为 **38.25%**（λ=0 为 38.15%）；B1/Deep 为 38.15/38.45%。R3 规则初值、operator、gate 为 77.7/51.4/59.5%，关系非最终状态 2.57%。fp32 固定面板中 R3 相关/无关编辑：ID 2/192 对 1/192，OOD 都 0/192。 | **单种子验证 screen 与小面板干预，不是三种子 OOD 功效测试。**状态标签有部分可解码性，却没有形成显著的答案因果路径；验证 gate 失败后停止。`config.json` 模板列出 3 种子，但实际 `selection.json` 只有种子 11 的三组，实际执行次数以 selection 为准。[配置](../results/evidence/state_diagnosis/state_supervised_full/config.json) · [选择](../results/evidence/state_diagnosis/state_supervised_full/selection.json) · [逐轮解码](../results/evidence/state_diagnosis/state_supervised_full/train_log.jsonl) · [λ=0 对照](../results/evidence/state_diagnosis/state_control6/selection.json) · [fp32 干预](../results/evidence/state_diagnosis/state_supervised_full/transplant_results_fp32.json) |

## 证据边界与后续使用

1. 旧的混合任务 OOD 已在多个阶段打开；Gold-H、Gold-Z1、probe、自由 `Z1` 辅助监督均应标为探索性诊断。新 hard-state、state-interface 和 support-hole 各有自己的测试与问题，不能拼成一条同数据消融曲线。
2. 明确区分三种“状态监督”：自由 `Z1` 的辅助解码、typed parser/转移的逐步 CE、以及 gold 状态直接注入。只有后两者把可验证状态放到实际递推计算的接口上；不同实验同时改变了表示与训练条件。
3. 可支持的主张是：在模板化单比特规则任务中，直接监督初值、指令与逐步状态，使相同单次读取系统从未学会任务转为正确的程序解析与状态执行。Gold 输入下执行器只需答案监督就可成功，当前未分离解析标签与累积状态标签的贡献。当解析器和监督相同时，二维连续、软反馈、硬反馈结果几乎相同。
4. 训练已覆盖全部局部转移，长题大量包含状态重置；当前长度成绩不等于在更大状态空间中维持长期依赖。规则任务的同监督接口对照未通过预定继续投入门槛，因此尚未运行关系/lookup 的同监督接口迁移。一般 JEV 或虚拟 CoT 能力仍未验证。
