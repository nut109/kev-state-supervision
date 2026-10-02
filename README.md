# Kev 的状态与指令监督实验

这个仓库研究如何把文本中的事实和规则训练成一个**可执行的状态接口**。最有用的实现是 R3-State：Kev 一次读取文本，预测初值与有序指令；一个共享小模块逐步更新状态，最终答案只由终态读出。训练时直接监督初值、指令和执行轨迹，测试时由模型预测它们。现有正结果来自自建的单比特布尔规则任务。

目前值得保留的是这套可复现的解析与执行训练流程。它把仅答案监督下未学会任务的模型训练成了能正确执行、并接受状态干预的系统。现有证据没有单独测出中间累积状态标签的贡献，因此本文用“状态与指令监督”指代完整标签包。

## 主要结果

原主实验从固定版本的 `jaredpalmer/kev-0.8b`（Qwen3.5-0.8B 基座＋Kev LoRA）加载并合并适配权重，弃用原指针头；训练最后六个主干块、初始值/算子/门解析头及共享的 770 参数状态转移。规则题训练深度 1–3，锁定 OOD 测试深度 4/5/6/8/10，各 200 题，按 world 隔离。以下为同一批 1,000 道 OOD 规则题、**一个训练种子**的结果：

| 模型 | 训练信号 | OOD 答案 | 深度 10 |
| --- | --- | ---: | ---: |
| R3 | 答案 CE | 50.4% | 50.5% |
| **R3-State** | 答案 CE＋初值/指令/逐步状态 CE | **99.4%** | **97.0%** |
| R3-IIT | 同上＋状态交换反事实答案 CE | 100% | 100% |
| Gold-State R3 | gold 初值与指令，执行器仅答案 CE | 100% | 100% |

R3-State 比同结构 answer-only R3 高 **49.0 个百分点**。这个差异同时体现在输入解析和执行轨迹中：

| 同一批 OOD 题的指标 | 仅答案 R3 | R3-State |
| --- | ---: | ---: |
| 完整初值与指令程序正确 | 0/1,000 | 977/1,000 |
| 完整状态轨迹正确 | 35/1,000 | 991/1,000 |
| 最终答案正确 | 504/1,000 | 994/1,000 |

R3-State 还在 589 次有效、非 donor-copy 状态替换中产生 584 次正确反事实答案。状态因而既符合标签语义，也参与实际答案计算；这个答案路径由架构预先规定。Answer-only 组在 64 题拟合检查中最高为 33/64，说明它首先遇到了学习与优化问题。Gold-State 执行器使用正确的初值与指令，**不加逐步状态损失**也达到 100%，所以当前实验支持完整监督流程有效，尚未证明逐步状态标签对小型执行器必需。[主实验逐题及指标](results/evidence/hard_iit/trial_r3_state_ood_test.json) · [答案监督对照](results/evidence/hard_iit/trial_r3_ood_test.json) · [gold 训练损失](kev/latent_iit_run.py)

R3-IIT 的 0.6 点增益在事后交叉组合中跟随解析器，未显示转移模块的独有改善。另一轮在相同规则任务和监督下冻结同一个 Kev 解析器，比较二维连续 logits、软反馈、硬反馈；三个配对种子的 OOD 均值为 **98.1% / 98.2% / 98.2%**。这两组结果使研究重点落在可监督的解析与状态执行流程上。[完整实验记录](docs/EXPERIMENTS.md)

## 成功任务的实际难度

这不是 XOR benchmark。规则包含 `copy`、`invert`、`and(gate)`、`or(gate)`，动态状态只有一位；只有没有常量重置的程序才归约为初值与 invert 奇偶性的 XOR。训练集已经覆盖全部 **12 种合法局部转移输入**。长度外推反复应用这些已见转移，静态指令序列仍随题目长度增长。

`and(false)` 与 `or(true)` 会擦除此前状态。对固定主测试面板的事后审计发现，深度 10 的 **187/200** 题包含这种重置；只有 13 题要求终态保留对初值的依赖，其中 R3-State 正确 12 题。十条规则的答案准确率因而不能单独证明十步持续依赖能力。逐题完整轨迹和状态干预是这套结果中更具体的执行证据。[任务审计与生成哈希](config/data_manifest.json) · [实际例题与状态监督含义](docs/METHOD.md)

已有成功范围是模板化的单比特规则解析与执行。关系链、lookup、真实检索问答、MuSiQue、ProofWriter 与一般 JEV/虚拟 CoT 能力尚未在这套流程上验证。

## 快速使用

需要 Python 3.12/3.13 与 [uv](https://docs.astral.sh/uv/)。以下命令在本仓库根目录执行。安装和 CPU 小测试不会下载 Kev 权重：

```bash
uv sync --group dev
uv run bash scripts/smoke_test.sh
```

独立生成原主实验的确定性合成数据：

```bash
uv run python -m kev.latent_iit_run prepare --out runs/data-only
```

完整复现入口会从 pinned Kev checkpoint 下载基座和适配权重，并依次训练 gold 正控制、单次读取解析器、Deep、answer-only R3、R3-State、R3-IIT。**这会消耗 GPU 资源；本次整理没有运行。**请先确认设备空间和使用计划，再执行：

```bash
uv run bash scripts/reproduce_main.sh runs/hard_iit cuda
uv run bash scripts/evaluate_main.sh runs/hard_iit cuda
```

`evaluate_main.sh` 只在全部验证检查点产生后评分 ID/OOD；原入口拒绝覆盖已有锁定测试结果。若只想确认小型 Support-Hole 程序可运行，`uv run python -m kev.latent_support_hole --smoke --out runs/support_hole/smoke.json` 在 CPU 上执行。完整五种子对照为 `uv run bash scripts/reproduce_support_hole.sh`。同监督接口实验可在取得主实验单次读取解析器检查点后运行：

```bash
uv run bash scripts/reproduce_interface.sh runs/hard_iit/onepass_rule_best.pt runs/state_interface cuda
```

接口脚本直接使用最终选定的 30 epoch 上限；历史开发版的 12 epoch 软反馈未通过短深度门槛，随后三组统一从相同初始化重训至 30 epoch，详见[实验总表](docs/EXPERIMENTS.md)。

主实验运行 0.8B 基座并更新最后六个块，需要 CUDA 训练环境；现存主实验归档没有足够的硬件日志来核准显卡型号、用时或显存峰值。Support-Hole 是不加载 Kev 的 CPU 自动机实验；同监督接口实验在 GPU 上缓存冻结解析器输出、在 CPU 上训练 770 参数转移模块。

## 仓库内容与复现边界

- `kev/latent_data.py`：世界隔离的数据生成、规则真值轨迹与干预 oracle；`kev/latent_iit_encoder.py`：一次读取的 Kev typed parser；`kev/latent_iit.py`：共享状态转移；`kev/latent_iit_run.py` 与 `kev/latent_iit_trials.py`：正控制、主训练和评价；`kev/latent_interface_run.py`：同监督接口对照；`kev/latent_support_hole.py`：独立 CPU 支持缺口实验。
- `results/summary.csv` 和 `summary.json` 给出有明确分母与来源键的机器可读结果；`results/evidence/` 保留必要的小型原始指标与诊断。[数据清单](config/data_manifest.json)记录划分、世界数和确定性生成的 JSONL 哈希，其中同监督接口划分与历史冻结哈希逐一匹配。历史 scratchpad、Read-Once、Gold-Z1 等**只有证据归档，没有打包全部旧训练代码**。
- 本包不包含基座/LoRA/训练检查点、完整生成数据或特征缓存。生成器可重建任务；原历史模型检查点只在原计算主机保留。主实验冻结时的 `latent_iit.py` 哈希与后来加入软/连续反馈的发布文件不同，默认硬路径保留，**不能宣称对旧 checkpoint 作逐字节重放**。详情见 [provenance.json](provenance.json)。
- 上游 Kev 代码遵循 Apache-2.0，归属与提取修改见 [NOTICE](NOTICE) 和 [LICENSE](LICENSE)。这里的研究结果不代表上游项目的官方结论。
