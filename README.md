# Kev 状态监督规则执行实验

这个仓库整理 Kev/JEV 决策模型上的**合成状态推理研究**。当前重点保留的完整方案是 R3-State：一次读取题目与规则，预测可验证的初始状态和指令，再用同一个小型转移模块逐步更新状态，并用中间状态标签训练整个路径。已验证任务是**自建布尔规则执行**；关系链、真实检索问答、MuSiQue、ProofWriter 和一般 JEV 推理能力均未在这套成功方案上验证。

## 主要结果

原主实验从固定版本的 `jaredpalmer/kev-0.8b`（Qwen3.5-0.8B 基座＋Kev LoRA）加载并合并适配权重，弃用原指针头；训练最后六个主干块、初始值/算子/门解析头及共享的 770 参数状态转移。规则题训练深度 1–3，锁定 OOD 测试深度 4/5/6/8/10，各 200 题，按 world 隔离。以下为同一批 1,000 道 OOD 规则题、**一个训练种子**的结果：

| 模型 | 训练信号 | OOD 答案 | 深度 10 |
| --- | --- | ---: | ---: |
| R3 | 答案 CE | 50.4% | 50.5% |
| **R3-State** | 答案 CE＋初值/指令/逐步状态 CE | **99.4%** | **97.0%** |
| R3-IIT | 同上＋状态交换反事实答案 CE | 100% | 100% |
| Gold-State R3 | gold 初值与指令，训练小型转移 | 100% | 100% |

R3-State 比同结构 answer-only R3 高 **49.0 个百分点**。Answer-only 组连 64 题拟合检查都没通过，因此这个差异说明该任务中显式状态/过程监督的完整训练方案有效；不能单独归因给循环共享或硬离散化。R3-IIT 的 0.6 点增益在事后交叉组合中跟随解析器，未显示转移模块的独有改善。Gold-State 是 oracle 输入对照，不是 Kev 解析文本的成绩。证据见[实验总表](docs/EXPERIMENTS.md)与[方法细节和具体例题](docs/METHOD.md)。

另一轮在**相同规则任务和监督**下，冻结同一个 Kev 解析器，比较二维连续 logits、软反馈、硬反馈。三个配对种子的 OOD 均值分别为 **98.1% / 98.2% / 98.2%**；深度 6/8/10 上反馈组只高 0.167 点，预设继续投入门槛未通过。这支持保留状态监督训练流程，并按任务需要选择方便的状态实现；它不能排除更宽的连续隐藏状态模型出现不同结果。

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
