# 状态监督的迭代规则执行：实际实现

本文对应原项目的 **R3-State**，即在一次文本读取之后，用受监督的二值状态反复执行布尔规则。主实验是自建合成规则任务，不是真实检索问答。跨实验结果见[实验总表](EXPERIMENTS.md)，主实验逐题记录见 [OOD 结果](../results/evidence/hard_iit/trial_r3_state_ood_test.json)；发布代码中的对应入口是 [`kev/latent_iit_trials.py`](../kev/latent_iit_trials.py)、[`kev/latent_iit_encoder.py`](../kev/latent_iit_encoder.py)、[`kev/latent_iit.py`](../kev/latent_iit.py)及 [`kev/latent_data.py`](../kev/latent_data.py)。

## 输入、状态与一次读取

每道题给出 `question`、顺序可能打乱的 `No`/`Yes` 选项，以及包含事实和规则的 `state` 文本。生成器另存初始真假值、规则算子、门值和执行轨迹，供训练标签与评价使用；这些私有字段**不进入** R3-State 的编码器。训练、测试时，模型从可见文本自行预测初始状态、每步算子和门值。

主实验以 bf16 加载 `jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e`，在加载时合并原 Kev 适配权重，使用其语言主干，弃用原有的选项指针头。最后六个主干块转为 fp32，与三个新线性头一起训练；其余主干冻结。一次前向读取完整的题目、选项和事实／规则文本：源文本中属于被问人物的规则行末位置分别预测四类算子 `copy/invert/and/or` 与三类门值 `none/false/true`，末尾位置预测二类初始真假。程序顺序、规则行数与选项的 `No`/`Yes` 含义由可见文本的结构确定；这个行定位是显式输入脚手架，不能描述成完全自由的端到端文本解析。编码器没有从生成器字段读取正确规则或初始状态。

令模型预测的初始二值状态为 \(z_0\)，第 \(t\) 条预测指令为 \((o_t,g_t)\)。算子、门值和状态均在前向计算中通过 argmax 变成 one-hot；训练反传使用 softmax 的 straight-through 梯度。共享转移模块把当前状态的 2 维 one-hot、算子的 4 维 one-hot 和门的 3 维 one-hot 拼成 9 维输入，经过 `Linear(9,64) → GELU → Linear(64,2)`，再硬化为 \(z_{t+1}\)。同一组 770 个转移参数用于所有步骤；每步只看当前状态和该步指令，不再读取原始文本。最多执行 10 步，达到该题可见规则数后不再更新。最终读出仅把终态与两个选项的真假含义匹配，不能直接读编码器表征或完整程序。

## 一个实际生成样本

下面是固定数据种子 `20260929`、训练划分中的 `causal-train-03281-b`，深度 3。原样本还含同一世界中 Person37 的事实和规则，供模型在完整文本中辨别被问对象；此处展示与问题直接相关的原文行：

```text
Question: Is Person52 ready after applying the rules for Person52?
Options: No | Yes
Facts:
Person52 has marker yellow.
Person52 is not verified.
Person52 is not bright.
Person52 is not available.
Person52 is not clear.
Rules:
For Person52: that subject is safe exactly when that subject is bright and is available.
For Person52: that subject is calm exactly when that subject is safe and is verified.
For Person52: that subject is ready exactly when that subject is calm or is clear.
```

生成器由原始事实得到 \(z_0=\text{false}\)，三步为 `and(false)`、`and(false)`、`or(false)`，所以轨迹是 `false → false → false → false`，答案为选项 `No`（索引 0）。`marker yellow` 不决定答案。轨迹、算子 ID 和门值 ID 是监督标签；在实际 R3-State 前向中，这些都由文本头预测，**不是** gold 注入。正对照 **Gold-State R3** 才把生成器的初始状态和指令作为输入，因此它不使用 Kev。

## 监督、训练与测试

生成器 [`solve_rules`](../kev/latent_data.py) 根据初始真假、四种算子及门值确定每步 gold 状态；[`pack_rules`](../kev/latent_iit_run.py)把这些原始字段转换成标签。R3-State 的答案损失是 \(A=\operatorname{CE}(\hat y,y)\)。状态损失 \(S\) 是以下四项交叉熵的等权平均：初始真假、实际步位的算子、实际步位的门值、以及**硬化前**下一状态 logits 对 gold 轨迹的预测。因此主实验使用 \(L=A+S\)；padding 步不计入过程损失。训练过程中，下一步始终接收本模型上一步预测的状态，没有用 gold 状态作 teacher forcing。R3 answer-only 使用同一可见输入、同一硬状态机器及初始化，只训练 \(L=A\)。Deep 也训练 \(A+S\)，但使用十个不同的绝对步位转移块。

主实验来自数据种子 `20260929` 的世界不交叠划分：规则题训练 5,000、验证 1,000、长深度 stress 100、ID 1,000、OOD 1,000；每个世界提供两道题。训练和 ID 深度为 1–3；OOD 深度为 4、5、6、8、10，各 200 题。四个文本输入组共用 seed 11 的新 Kev／头／转移初始化、相同事实样本顺序、batch 8 和四轮训练上限。AdamW 权重衰减 0.01；主干、三个头、转移模块的学习率分别为 `2e-5`、`3e-4`、`3e-3`。按验证集答案准确率、再按答案 NLL 选择检查点；R3-State 入选 epoch 1。主实验 R3-State 检查点约有 122,936,715 个可训练参数，含主干可训练块；770 只是循环模块的参数数。

正式测试仍由 Kev 预测初始值与所有指令；gold 轨迹仅用于评分。R3-State 在 1,000 道 OOD 规则题上为 **994/1,000（99.4%）**，深度 10 为 **194/200（97.0%）**；answer-only R3 为 **504/1,000（50.4%）**。这组差异表明在此训练设置和合成任务上，显式过程监督使完整系统学会执行；它不单独证明循环或硬离散化比所有可比架构更强。Gold-State R3 的 **1,000/1,000** 是 oracle 输入上限，不是 Kev 从文本获得状态的成绩。各臂的结果与分母分别在[状态监督](../results/evidence/hard_iit/trial_r3_state_ood_test.json)、[仅答案](../results/evidence/hard_iit/trial_r3_ood_test.json)及[oracle](../results/evidence/hard_iit/trial_gold_ood_test.json)记录中。

同一世界的两道题还用于不参与 R3-State 训练的因果测试：在合法步位只把 donor 的**模型预测动态状态**放进 recipient，继续执行 recipient 自己剩余的规则、门值和选项，并与符号执行器的反事实答案比较。OOD 中，能改变答案且不只是复制 donor 答案的 589 次替换有 **584 次**反事实答案一致、**578 次**出现正确方向翻转；5,334 次无效状态替换中有 **15 次**导致答案翻转。这说明在所施加的接口下该状态确实控制输出；由于最终读出被架构限定为只看终态，不能据此声称网络自发发现了这种接口。[干预计数](../results/evidence/hard_iit/trial_r3_state_ood_test.json)

## 同监督接口对照

后续对照使用新的数据种子 `20261001`，冻结先前训练的单次读取 Kev 解析器，把**同一批预测的**初始状态、算子和门 logits 缓存给三个 770 参数转移模型。A 在步间传递二维原始状态 logits，B 传递二维 softmax，C 传递 straight-through 硬 one-hot；三组答案读出都看硬化的终态。三组均用答案 CE 加 `0.25 ×` 每步硬化前状态 CE，无 IIT、无 teacher forcing；seed 为 11、23、37。短深度验证过门后，锁定 OOD 的 B/C 比 A 在深度 6/8/10 仅高 **0.167 个百分点**，配对区间为 `[0, 0.389]` 个百分点，B 与 C 的逐题答案相同。冻结解析器的完整程序预测只有 890/1,000 精确，但这些预测交给符号执行器后答案为 982/1,000；B/C 每题都匹配后者，说明该面板的最终答案已受解析误差限制。因此该任务上没有识别出硬反馈独有收益。A 只是**二维连续 logits**，不是任意宽的隐状态或可绕开状态解码的网络。[接口对照逐题及区间](../results/evidence/state_interface/locked_results.json)

## 历史实现与发布代码的边界

本目录从原项目提取了数据生成、Kev 解析器、硬状态机器及训练／评价入口；对应的原始路径分别是 `kev/kev/latent_data.py`、`latent_iit_encoder.py`、`latent_iit.py`、`latent_iit_run.py` 和 `latent_iit_trials.py`。主实验冻结清单把当时的 `latent_iit.py` 记为 SHA-256 `9c5e86a4…`；在同监督接口实验中，该文件加入 A/B 反馈选项后变为 `d5ff2b8c…`。发布的默认 `feedback="hard"` 保留主实验的硬状态计算路径，但该文件**不与历史冻结文件逐字节相同**。发布包也不包含大型训练权重和原始完整数据；历史清单只保留旧检查点哈希。故仓库可以从固定生成器和已注明的 Kev 起点重新运行方案，不能宣称凭本包对历史检查点做精确字节级重放。参见[历史冻结清单](../results/evidence/hard_iit/frozen_manifest.json)与[发布来源说明](../provenance.json)。
