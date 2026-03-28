# CT-Agent 项目概览

本文档基于当前仓库代码、配置、测试和 `results/` 中的实验产物整理，目标是说明这个项目已经完成了什么、代码是如何组织的，以及当前哪些部分已经落地、哪些部分仍是接口或占位实现。

## 1. 项目在做什么

这是一个面向 CT 图像的 Agent 系统，主线是：

`CT 图像 -> 退化感知 -> Planner 决策 -> 多工具修复 -> 下游诊断/评估`

它不是单一的图像增强脚本，而是把以下几件事串成了一个整体：

1. 读取 CT 数据并做基础预处理。
2. 自动判断图像退化类型和严重程度。
3. 用规则或 LLM 生成修复工具链。
4. 调用经典工具、学习式工具或 MCP-style 工具完成修复。
5. 用 IQA、Judge 和闭源 API 诊断任务评估修复是否有效。
6. 在 CQ500 等数据上跑批量实验并保存结果。

## 2. 你已经完成的工作

从当前仓库内容来看，已经完成的工作主要有以下几类。

### 2.1 已跑通的最小可行主流程

入口脚本 `scripts/run_pipeline.py` 已经实现了一个清晰的 MVP 流程：

1. 读取配置和输入图像。
2. 用 `DegradationDetector` 做退化检测。
3. 用 `RuleBasedPlanner` 生成工具调用计划。
4. 用 `Executor` 串行执行工具链。
5. 将最终修复结果写出到输出目录。

这说明项目最基础的 "detect -> plan -> execute" 主链路已经打通。

### 2.2 感知、规划、执行三层架构已经成型

#### 感知层

`src/tools/mcp_style/` 下已经实现了 4 个面向 LLM/Agent 的结构化工具：

- `analysis_tool.py`: 退化类型分析。
- `perception_tool.py`: IQA 与感知信息提取。
- `statistics_tool.py`: 统计量和分布信息提取。
- `restoration_tool.py`: 修复执行与前后对比输出。

这些工具说明你不只是做了普通函数，而是已经在往 "MCP-style 工具接口" 方向抽象。

#### 规划层

`src/planner/` 已经形成多种 planner 路线：

- `rule_planner.py`: 规则式 planner，适合作为 MVP 和稳定基线。
- `agent_based.py`: 基于 LLM 的 planner，会先调用 perception tools，再把结构化结果交给 `llm/planner_caller.py` 生成计划。
- `policy_rl_placeholder.py`: 预留了 RL planner 接口。

其中 `AgentBasedPlanner` 已经支持：

- perception -> LLM planning 主流程；
- 兼容旧版 `api_caller`；
- 无 LLM 时回退到规则规划。

说明你的规划层已经从单纯规则系统扩展为多策略统一接口。

#### 执行层

`src/executor/engine.py` 已经完成执行引擎：

- 按 `Plan` 顺序调用工具；
- 缓存工具实例；
- 记录每一步参数、耗时、成功与否、图像尺寸等信息；
- 通过 `last_trace` 暴露本次执行轨迹。

这意味着你的执行层不仅能跑工具，还具备可追踪、可评估、可供 Judge/Memory 使用的基础设施。

### 2.3 工具体系已经比较完整

#### 经典工具

`src/tools/classical/` 下已经有较完整的传统图像处理工具集合，包括：

- 去噪：`denoise.py`、`wavelet.py`、`median.py`、`bm3d_denoise.py`
- 去模糊：`deblur.py`
- 增强与锐化：`enhance.py`、`sharpen.py`
- 直方图相关：`histogram.py`
- 修补：`inpaint.py`
- 裁剪与伪影相关处理：`clip.py`、`mar.py`

#### 学习式工具

`src/tools/learned/` 下已经接入或预留了学习式工具：

- `dncnn_tool.py`: 已用于实际实验。
- `mar_adapter.py`
- `sr_adapter.py`

从结果文件看，`DnCNN` 已经被真实训练并纳入评测，而 `MAR/SR adapter` 仍偏占位或降级实现。

#### 工具注册机制

`src/tools/registry.py` 已经把工具创建做成统一入口，这对于 planner 和 executor 解耦很关键。

### 2.4 数据、退化模拟与数据工程已经做了不少

仓库里不仅有 `src/datasets/`，还有单独的 `dataset/` 工程目录，说明你把 "训练/实验数据构建" 和 "运行时数据集封装" 做了分层。

#### `src/datasets/`

- `ct_dataset.py`
- `paired_dataset.py`

偏向训练/评估阶段直接消费的数据集抽象。

#### `dataset/`

- `dicom_scanner.py`、`dicom_reader.py`、`ct_slice_exporter.py`
- `manifest.py`
- `degradation_builder.py`
- `toy.py`
- `mar/` 子目录中的一整套 MAR 数据生成与物理仿真代码

其中 `dataset/mar/` 已经覆盖：

- CT 几何建模
- 组织分解
- 能量转换
- 正弦图处理
- MAR 仿真
- CQ500 读取
- 数据集构建

这说明你不只是做了推理流程，也做了实验数据构建与退化合成管线。

### 2.5 下游诊断接口已经接入

`src/downstream/closed_api_adapter.py`、`llm/diagnosis_caller.py`、`llm/api_client.py` 说明你已经把闭源 API 诊断纳入系统。

当前代码已支持几种不同的评测模式：

1. 直接对 clean/degraded 图像做 API 诊断。
2. 先修复再做 restored 图像诊断。
3. 在更正式的闭环实验里比较 rule、API-text、API-vision 等不同策略。

这一步非常关键，因为它把项目从 "图像修复" 扩展到了 "修复是否改善下游医学判断"。

### 2.6 评估体系已经落地

项目里有两类评估代码：

#### 通用评估

`src/eval/evaluator.py` 负责通用的图像评估，比如配对数据上的 PSNR/SSIM 等指标。

#### 数据集/实验专用评估

根目录 `eval/` 里已经有：

- `cq500_iqa_restore_eval.py`
- `cq500_api_eval.py`
- `cq500_manifest.py`
- `cq500_labels.py`
- `metrics.py`
- `bhx_loader.py`

这说明你不仅有通用 evaluator，还为 CQ500 这类真实实验写了专门的评测逻辑、标签加载和病例组织代码。

### 2.7 Judge、Memory、闭环 Replan 机制已经搭起来

除了简单单次修复，你还做了闭环系统的关键部件：

- `src/judge/quality_judge.py`
- `judge/safety_judge.py`
- `pipeline/replan.py`
- `pipeline/agent_loop.py`
- `memory/store.py`
- `memory/experience.py`

说明项目已经具备以下能力：

1. 对修复质量进行判定。
2. 对安全性/稳定性进行判定。
3. 根据判定结果进行 replan。
4. 记录经验或执行轨迹，支撑后续分析或学习。

### 2.8 脚本、训练和测试体系已经基本齐备

#### 脚本

`scripts/` 下已经覆盖多种用途：

- 主流程：`run_pipeline.py`
- 通用评估：`evaluate.py`
- 退化模拟：`simulate_degradation.py`
- CQ500 IQA 评测：`run_cq500_iqa_eval.py`
- CQ500 API 评测：`run_cq500_api_eval.py`
- 正式比较实验：`run_formal_comparison.py`
- 工具集比较：`run_toolset_comparison.py`
- 闭环 demo：`run_closed_loop_demo.py`
- toy workflow：`run_toy_workflow.py`
- 数据构建：`build_clean_dataset.py`、`build_degraded_dataset.py`、`build_mar_dataset.py`
- 训练：`train_dncnn.py`

#### 测试

`tests/` 下已有：

- `test_io.py`
- `test_degradation.py`
- `test_tools.py`
- `test_planner.py`
- `test_mcp_tools.py`
- `test_agent_planner.py`
- `test_downstream_api.py`
- `test_iqa.py`
- `test_rl.py`

这说明项目不是只写了实验脚本，也已经开始补单元测试和模块级验证。

## 3. 当前代码结构解释

下面按照职责解释仓库结构。

```text
ctagent/
├── configs/        # 配置系统，管理实验、planner、tools、data、model
├── src/            # 核心库代码
├── dataset/        # 数据工程与退化/MAR 构建代码
├── eval/           # CQ500/BHX 等实验评测逻辑
├── llm/            # LLM/API 客户端、prompt、解析、planner caller
├── pipeline/       # 单次和闭环 pipeline 编排
├── scripts/        # 各类运行入口
├── tests/          # 单元测试和集成测试
├── results/        # 已完成实验输出
├── checkpoints/    # 训练权重、数据划分等中间产物
├── docs/           # 项目文档
└── try_output/     # 可视化和临时输出
```

### 3.1 `configs/`

职责是把项目从 "写死在代码里" 变成 "可切换实验设置"。

常见子目录：

- `configs/data/`: 数据路径、数据集配置、MAR 仿真配置等。
- `configs/degradation/`: 退化相关配置。
- `configs/planner/`: planner 配置。
- `configs/tools/`: 工具参数配置。
- `configs/model/`: 模型训练配置，比如 DnCNN。
- `configs/experiment/`: 不同实验场景的入口配置。

`configs/experiment/minimal.yaml` 对应最小可行链路，是最适合理解整个系统的入口配置。

### 3.2 `src/`

这是最核心的业务代码区。

#### `src/io/`

负责 CT 读写和窗宽窗位处理，是整个项目的数据入口层。

#### `src/degradations/`

负责退化类型定义、检测和模拟。系统的规划起点依赖这里输出的 `DegradationReport`。

#### `src/tools/`

负责所有可执行工具及其统一注册：

- `classical/`: 传统图像处理工具。
- `ct/`: CT-specific 工具接口。
- `learned/`: 学习式工具。
- `mcp_style/`: 给 LLM/Agent 使用的结构化工具。

#### `src/planner/`

负责把感知结果转换为行动计划：

- 规则规划
- LLM 规划
- RL 规划接口

#### `src/executor/`

负责真正执行计划，并输出执行 trace。

#### `src/downstream/`

负责把修复后的图像接到下游诊断或分类任务上。

#### `src/judge/`

负责从质量、可接受性等角度对一次修复结果做打分或判定。

#### `src/eval/`

负责通用指标评估与结果聚合。

#### `src/rl/`

负责 RL 环境、奖励、轨迹和 verl 对接接口，目前更像是为后续扩展预留的骨架。

### 3.3 `dataset/`

这里更偏向离线数据工程，而不是推理时的核心库。

它承担的任务包括：

1. 扫描 DICOM。
2. 导出切片。
3. 构建 manifest。
4. 合成退化数据。
5. 构建 MAR 实验数据。

如果说 `src/datasets/` 是训练/评估阶段的数据读取接口，那么 `dataset/` 更像数据准备流水线。

### 3.4 `eval/`

根目录 `eval/` 不是重复代码，而是偏实验层的专用评估逻辑。

这里主要做：

1. CQ500 病例和切片组织。
2. 标注读取。
3. BHX 辅助信息加载。
4. 批量运行诊断评测。
5. 汇总统计和产出结果文件。

### 3.5 `llm/`

这里承接与外部大模型交互的逻辑：

- API client
- planner caller
- diagnosis caller
- prompt builder
- response parser

这部分把上层业务和具体 LLM 服务解耦了。

### 3.6 `pipeline/`

这里负责把 detector、planner、executor、judge 串成完整流程。

目前至少有两条主要形态：

- `single_pass.py`: 单次流程
- `agent_loop.py`: 闭环多轮流程

此外还有：

- `replan.py`
- `api_guided_planner.py`
- `types.py`

说明 pipeline 已经不只是单次调用，而是朝着可反思、可重试、可对比的 Agent workflow 发展。

## 4. 端到端流程怎么串起来

### 4.1 最简单的一条链路

`scripts/run_pipeline.py`

```text
输入 CT 图像
  -> src.io 读取
  -> src.degradations 检测退化
  -> src.planner.rule_planner 生成计划
  -> src.executor.engine 执行工具链
  -> src.io 写出修复图像
```

这条链路适合做基线和本地验证。

### 4.2 LLM 增强的感知-规划链路

`src/planner/agent_based.py`

```text
输入图像
  -> AnalysisTool / PerceptionTool / StatisticsTool
  -> llm/planner_caller.py
  -> 输出 Plan
```

也就是说，你的 LLM planner 不是盲目直接看图，而是先拿到结构化感知结果，再做计划生成。

### 4.3 CQ500 修复评测链路

`scripts/run_cq500_iqa_eval.py`

```text
构建 CQ500 manifest
  -> 读取 test split 防泄漏
  -> 选择 rule / llm / both
  -> 对每个病例切片执行 restoration
  -> 统计 PSNR / SSIM / tool usage / degradation distribution
  -> 保存到 results/cq500_iqa_eval/
```

### 4.4 CQ500 API 诊断评测链路

`scripts/run_cq500_api_eval.py`

```text
构建带标签的 CQ500 manifest
  -> 选择 clean / degraded / restored 输入类型
  -> 调用闭源 LLM/VLM API 诊断
  -> 聚合 accuracy / F1 / latency 等结果
  -> 保存到 results/cq500_api_eval/
```

### 4.5 正式比较实验链路

`scripts/run_formal_comparison.py`

这个脚本已经把多种 pipeline 放到同一张对照表里，包括：

- `rule-sp`
- `rule-cl`
- `api-mock`
- `api-text`
- `api-vision`

说明你已经开始做系统级消融与正式对比，而不是只看单点指标。

## 5. 从现有结果看，已经完成了哪些实验

### 5.1 CQ500 IQA 修复评测已经跑出结果

从 `results/cq500_iqa_eval/eval_log.txt` 可以看出：

- 训练集 311 个 case，测试集 77 个 case。
- 当前展示的评测覆盖 12 个测试 case，共 120 次 eval。
- 已显式做 train/test split，并检查了 `Data leakage: NONE`。

规则 planner 与 LLM planner 的恢复结果已经产出：

- `rule`: PSNR `14.17 -> 43.66`，提升 `+29.49`；SSIM `0.6403 -> 0.9846`，提升 `+0.3443`
- `llm`: PSNR `14.17 -> 38.99`，提升 `+24.81`；SSIM `0.6403 -> 0.9727`，提升 `+0.3324`

同时已经统计了工具使用情况：

- `clip_extreme`: 119 次
- `denoise_dncnn`: 119 次
- `inpaint_biharmonic`: 55 次
- `denoise_wavelet`: 4 次

这表明：

1. IQA 评测链路已经完整跑通。
2. rule 和 llm 两种 planner 已经做了真实对比。
3. DnCNN 已经实打实进入修复主链路。

### 5.2 CQ500 API 诊断评测已经跑出 clean vs degraded 对比

`results/cq500_api_eval/summary.csv` 和 `summary.json` 表明：

- `clean` 平均准确率约 `0.8383`
- `degraded` 平均准确率约 `0.8246`
- `degraded` 相对 `clean` 有约 `1.63%` 的准确率下降

更重要的是，`clean` 与 `degraded` 的宏平均 F1 差别很明显：

- `clean macro_f1 = 0.2183`
- `degraded macro_f1 = 0.0991`

这说明你已经把 "图像退化会不会伤害下游诊断" 这件事量化了出来。

### 5.3 正式比较实验已经跑出多策略对照

`results/formal_comparison/summary.csv` 显示你已经完成至少 5 种策略的正式对比：

- `rule-sp`
- `rule-cl`
- `api-mock`
- `api-text`
- `api-vision`

当前结果中：

- `api-mock` 的 diagnosis accuracy 为 `0.60`
- `rule-sp` / `rule-cl` 为 `0.55`
- `api-text` / `api-vision` 为 `0.50`

同时 `api-vision` 具有更高的工具多样性，说明你已经把对比从 "是否更准" 扩展到 "是否更会用工具、是否更会 replan"。

## 6. 哪些部分已经落地，哪些还是占位

### 6.1 已经比较成熟的部分

- 基础 CT I/O
- 退化检测
- 规则 planner
- Agent-based planner 主流程
- 工具注册和执行引擎
- 经典工具链
- DnCNN 工具接入与评测
- CQ500 IQA 实验
- CQ500 API 诊断实验
- 闭环 pipeline 和正式比较实验
- 基础测试集

### 6.2 明显还在预留/占位的部分

从代码结构和命名看，以下模块更像后续扩展点：

- `src/planner/policy_rl_placeholder.py`
- `src/rl/verl_adapter.py`
- `src/tools/ct/` 中部分 CT-specific 深度模型接口
- `src/tools/learned/mar_adapter.py`
- `src/tools/learned/sr_adapter.py`
- `src/downstream/classifier.py`

也就是说，你的系统骨架已经很完整，但 RL 策略学习、部分 CT 专用深度恢复模型、本地分类器这些方向还处在预留阶段。

## 7. 代码阅读建议

如果后续你想快速给别人介绍项目，推荐按下面顺序阅读：

1. `README.md`
2. `scripts/run_pipeline.py`
3. `src/degradations/`
4. `src/planner/rule_planner.py`
5. `src/planner/agent_based.py`
6. `src/executor/engine.py`
7. `src/tools/mcp_style/`
8. `scripts/run_cq500_iqa_eval.py`
9. `scripts/run_cq500_api_eval.py`
10. `results/` 中的实验结果

这样可以先理解骨架，再理解实验层。

## 8. 一句话总结

当前这个仓库已经不是一个简单的 CT 图像增强 demo，而是一个已经具备以下特征的系统原型：

- 有完整主链路；
- 有多种 planner；
- 有成体系工具库；
- 有真实数据实验；
- 有下游诊断评测；
- 有闭环对比实验；
- 有继续扩展到 RL 和更强 CT 专用模型的接口。

如果用阶段来描述，你现在已经完成了 "可运行、可评测、可对比、可扩展" 的第一版系统，而不是只停留在想法或单脚本验证阶段。
