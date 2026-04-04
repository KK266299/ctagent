# CT-Agent 代码库详解

## 项目概述

**项目名称**: CT-Agent-Frontdoor

**版本**: 0.1.0

**核心思想**: 一个 **感知-规划-执行** 的智能体系统，用于 CT 图像退化分析与多工具修复，并评估修复后图像对下游医学诊断的影响。

```
CT 图像 → 退化检测 → 规划修复方案 → 执行修复工具 → 质量评判 → 下游诊断评估
```

---

## 目录结构总览

```
ctagent/
├── src/                    核心模块
│   ├── io/                 CT 图像读写
│   ├── datasets/           数据集加载
│   ├── degradations/       退化检测与模拟
│   ├── iqa/                图像质量评估 (IQA)
│   ├── tools/              修复工具生态系统
│   │   ├── classical/      经典图像处理工具
│   │   ├── ct/             CT 专用工具
│   │   └── mcp_style/      MCP 风格工具（供 LLM 调用）
│   ├── planner/            规划器（规则/LLM/RL）
│   ├── executor/           执行引擎
│   ├── judge/              质量与安全评判
│   ├── eval/               流水线评估
│   ├── downstream/         下游诊断任务
│   ├── rl/                 强化学习接口
│   └── utils/              工具函数
│
├── llm/                    LLM API 层
│   ├── api_client.py       统一 API 客户端
│   ├── planner_caller.py   规划 LLM 调用
│   ├── diagnosis_caller.py 诊断 LLM 调用
│   ├── prompt_builder.py   提示词构建
│   └── response_parser.py  响应解析
│
├── eval/                   CQ500 评估专用模块
│   ├── cq500_labels.py     GT 标签加载
│   ├── cq500_manifest.py   评估清单构建
│   ├── cq500_api_eval.py   API 评估核心
│   └── metrics.py          多标签分类指标
│
├── pipeline/               高层流水线编排
│   ├── single_pass.py      单次执行流水线
│   ├── replan.py           重规划策略
│   └── api_guided_planner.py  API 引导规划
│
├── executor/               执行追踪
│   └── trace.py            执行轨迹记录
│
├── downstream/             下游任务
│   └── mock_diagnosis.py   本地 CNR 模拟诊断
│
├── memory/                 经验存储
│   ├── experience.py       经验记录数据结构
│   └── store.py            JSON 持久化存储
│
├── judge/                  评判模块
│   ├── safety_judge.py     安全性评判
│   └── base.py             评判基类
│
├── configs/                配置文件
│   ├── data/               数据路径配置
│   ├── degradation/        退化模型配置
│   ├── tools/              工具参数配置
│   ├── planner/            规划器配置
│   └── experiment/         实验配置
│
├── scripts/                可运行脚本
├── tests/                  单元测试
├── data/                   数据存储
├── dataset/                数据集文件
└── results/                输出结果
```

---

## 核心模块详解

### 1. `src/io/` — CT 图像读写

| 文件 | 功能 |
|------|------|
| `readers.py` | 多格式读取（DICOM、NIfTI、PNG、NumPy），自动转换为 HU 值 |
| `writers.py` | 多格式写入，支持 DICOM 转换 |
| `windowing.py` | CT 窗位窗宽预设（软组织窗、肺窗、骨窗、脑窗、肝窗） |

**关键函数**: `read_ct()`, `write_ct()`, `apply_window()`

---

### 2. `src/datasets/` — 数据集加载

| 类 | 功能 |
|----|------|
| `CTDataset` | 单图像加载，用于推理/退化检测 |
| `PairedCTDataset` | 退化/原始图像配对加载，用于修复评估 |

支持 DICOM、NIfTI、PNG、NumPy 格式，确保配对一一对应。

---

### 3. `src/degradations/` — 退化检测与模拟

#### 退化类型枚举 (`DegradationType`)

| 类型 | 说明 |
|------|------|
| `NOISE` | 噪声 |
| `BLUR` | 模糊 |
| `ARTIFACT_METAL` | 金属伪影 |
| `ARTIFACT_STREAK` | 条纹伪影 |
| `ARTIFACT_RING` | 环形伪影 |
| `LOW_RESOLUTION` | 低分辨率 |
| `LOW_DOSE` | 低剂量 |
| `UNKNOWN` | 未知 |

#### 严重程度 (`Severity`): MILD / MODERATE / SEVERE

#### 关键类

| 类 | 功能 |
|----|------|
| `DegradationDetector` | 基于 IQA 阈值的退化检测（使用拉普拉斯 MAD 噪声估计） |
| `DegradationSimulator` | 合成退化注入（高斯噪声、高斯模糊、下采样） |

**输出**: `DegradationReport`（退化列表 + IQA 分数 + 元数据）

---

### 4. `src/iqa/` — 图像质量评估

| 模块 | 指标 |
|------|------|
| `metrics.py` | PSNR（峰值信噪比）、SSIM（结构相似性）、批量计算 |
| `no_reference.py` | 无参考指标：清晰度 (`sharpness`)、噪声估计 (`noise_estimate`) |

---

### 5. `src/tools/` — 修复工具生态系统

这是项目的核心部分，包含三层工具架构：

#### 工具基础设施

```python
# 所有工具继承 BaseTool
class BaseTool(ABC):
    name: str           # 工具名称
    description: str    # 功能描述
    def run(image, **kwargs) -> ToolResult  # 执行修复
```

`ToolRegistry` 提供装饰器注册和按名查找功能。

#### 经典工具 (`src/tools/classical/`)

| 工具 | 类 | 适用场景 |
|------|-----|---------|
| 高斯去噪 | `GaussianDenoise` | 快速去噪，会模糊边缘 |
| 双边滤波去噪 | `BilateralDenoise` | 保边去噪，适合轻中度噪声 |
| 全变分去噪 | `TVDenoise` | 强保边能力，适合中重度噪声 |
| 非局部均值去噪 | `NLMDenoise` | 基于自相似的结构感知去噪 |
| 维纳滤波去噪 | `WienerDenoise` | 频域最优，适合白噪声 |
| CLAHE | `CLAHE` | 对比度受限自适应直方图均衡 |
| 反锐化掩模 | `UnsharpMask` | 轻度模糊图像的锐化 |

#### CT 专用工具 (`src/tools/ct/`)

| 工具 | 功能 | 状态 |
|------|------|------|
| `MARTool` | 金属伪影去除 (MAR) | 占位符，待接入 RISE-MAR |
| `LDCTDenoiseTool` | 低剂量 CT 去噪 | 占位符，待接入深度学习模型 |
| `CTSuperResolutionTool` | CT 超分辨率 | 占位符 |

#### MCP 风格工具 (`src/tools/mcp_style/`)

为 LLM 智能体设计，返回结构化 JSON 而非图像：

| 工具 | 功能 |
|------|------|
| `AnalysisTool` | CT 退化分析 → JSON 报告（退化类型、严重度、IQA） |
| `PerceptionTool` | IQA 指标计算（无参考 + 全参考） |
| `RestorationTool` | 包装 BaseTool，返回修复前后对比指标 |
| `StatisticsTool` | HU 分布统计（均值、标准差、百分位、熵、峰度） |

---

### 6. `src/planner/` — 规划器

三种规划策略：

#### a) 规则规划器 (`RuleBasedPlanner`)

硬编码的 `(退化类型, 严重度) → 工具序列` 映射表。

```
例: (NOISE, MODERATE) → [BilateralDenoise]
    (ARTIFACT_METAL, SEVERE) → [MARTool, TVDenoise]
```

#### b) 智能体规划器 (`AgentBasedPlanner`)

LLM 驱动的 感知→规划→执行 循环：
1. 调用 AnalysisTool + PerceptionTool + StatisticsTool 收集感知信息
2. 将感知结果发送给 LLM 请求生成修复方案
3. LLM 不可用时回退到规则规划器

#### c) RL 策略规划器 (`RLPolicyPlanner`)

基于强化学习的规划（占位符，当前回退到规则规划）。

**输出**: `Plan`（有序的 `ToolCall` 列表 + 推理说明）

---

### 7. `src/executor/` — 执行引擎

`Executor` 按顺序执行 Plan 中的工具链：

```
Plan(tools=[T1, T2, T3]) + image
    → T1(image) → result1
    → T2(result1.image) → result2
    → T3(result2.image) → result3
    → list[ToolResult] + ExecutionTrace
```

`ExecutionTrace` 记录每步的耗时、成功/失败、参数等调试信息。

---

### 8. `src/judge/` — 质量评判

| 评判方式 | 方法 | 指标 |
|----------|------|------|
| 有参考 | `judge_with_reference()` | PSNR/SSIM 是否超过阈值 |
| 无参考 | `judge_no_reference()` | 综合分 = 噪声降低 + 干净度 + 结构保持 |

**输出**: `JudgeVerdict`（通过/分数/原因/详情）

---

### 9. `src/downstream/` — 下游诊断任务

#### 闭源 API 适配器 (`ClosedAPIAdapter`)

对接 GPT-4o、Claude 等视觉语言模型进行 CT 诊断：

| 方法 | 功能 |
|------|------|
| `predict()` | 直接诊断 |
| `predict_with_tools()` | 工具增强诊断（先提供 IQA 等上下文） |
| `compare_diagnosis()` | 对比原图/退化图/修复图的诊断结果 |

#### 模拟诊断 (`MockDiagnosis`)

本地基于 CNR（对比噪声比）的病灶检测，无需 LLM：
- 高噪声 → 高局部标准差 → 低 CNR → 漏检
- 用于工作流验证

---

### 10. `src/rl/` — 强化学习接口

| 模块 | 功能 |
|------|------|
| `env.py` | Gym 风格环境：动作空间 = 所有注册工具 + STOP |
| `reward.py` | 多维奖励：IQA 改善 + 诊断一致性 + 效率惩罚 |
| `trajectory.py` | 轨迹记录（Transition → Trajectory → TrajectoryBuffer） |
| `verl_adapter.py` | verl RL 框架桥接（占位符） |

---

## 高层流水线 (`pipeline/`)

### 单次执行流水线 (`SinglePassPipeline`)

```
输入图像
  │
  ▼
退化检测 (DegradationDetector)
  │ → DegradationReport
  ▼
规划 (Planner)
  │ → Plan [tool1, tool2, ...]
  ▼
执行 (Executor)
  │ → list[ToolResult] + ExecutionTrace
  ▼
质量评判 (QualityJudge)
  │ → JudgeVerdict (通过/失败)
  ▼
下游诊断 (ClosedAPIAdapter / MockDiagnosis)
  │ → DiagnosisResult
  ▼
生成报告 (SinglePassResult)
```

### 重规划流水线 (`ReplanPipeline`)

当 QualityJudge 判定失败时，触发重规划：
1. 收集失败原因 + 当前质量指标
2. LLM 生成新方案或调整参数
3. 重新执行（最多 N 次迭代）

---

## LLM API 层 (`llm/`)

### 统一客户端架构

```
BaseLLMClient (抽象基类)
    ├── OpenAIClient     (OpenAI / OpenRouter / 本地兼容)
    └── AnthropicClient  (Anthropic Claude)
```

### 专用调用器

| 调用器 | 功能 |
|--------|------|
| `PlannerCaller` | 规划调用：分析+感知+统计 → Plan JSON |
| `DiagnosisCaller` | 诊断调用：图像 → 诊断结果 JSON |

### 响应解析

- `parse_plan_json()`: 从 LLM 响应解析 Plan
- `parse_guided_decision()`: 解析 API 引导的规划决策
- `PARAM_RANGES` + `clip_params()`: 工具参数校验与裁剪

---

## 经验存储 (`memory/`)

| 类 | 功能 |
|----|------|
| `ExperienceRecord` | 单次流水线执行的完整记录（退化信息、工具序列、评判结果等） |
| `ExperienceStore` | JSON 文件持久化存储，支持按退化类型查询、查找最佳/失败路径 |

用于少样本规划和强化学习数据收集。

---

## CQ500 评估模块 (`eval/`)

独立于主流水线的 VLM 诊断评估系统，详见 [CQ500 评估流程图](cq500_api_eval_flow_zh.md)。

核心目的：在 CQ500 数据集上对比 VLM 在 clean / degraded / restored 图像上的诊断性能。

---

## 可运行脚本 (`scripts/`)

| 脚本 | 用途 |
|------|------|
| `run_pipeline.py` | 主入口：配置 → 退化检测 → 规划 → 执行 |
| `evaluate.py` | 配对数据集批量评估 |
| `simulate_degradation.py` | 合成退化注入 |
| `run_cq500_api_eval.py` | CQ500 VLM 诊断评估 |
| `run_toy_workflow.py` | 带合成标签的玩具案例 |
| `run_formal_comparison.py` | 系统化工具组合对比 |
| `run_toolset_comparison.py` | 工具性能对比 |
| `test_mar_pipeline.py` | MAR 专项测试 |
| `visualize_mar_results.py` | MAR 结果可视化 |
| `build_clean_dataset.py` | 数据集准备 |
| `build_degraded_dataset.py` | 退化图像合成 |
| `build_mar_dataset.py` | MAR 数据集构建 |
| `run_closed_loop_demo.py` | 闭环智能体演示 |

---

## 测试 (`tests/`)

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_io.py` | DICOM/NIfTI/PNG 读写 |
| `test_degradation.py` | 退化检测与模拟 |
| `test_iqa.py` | IQA 指标计算 |
| `test_tools.py` | 单个修复工具 |
| `test_planner.py` | 规则规划器 |
| `test_agent_planner.py` | LLM 规划器 |
| `test_mcp_tools.py` | MCP 风格工具链 |
| `test_downstream_api.py` | API 诊断调用 |
| `test_rl.py` | RL 环境与奖励 |

---

## 开发路线图

| 版本 | 重点 |
|------|------|
| **v0.1** | MVP：规则规划器 + 经典工具 + IQA 评估 |
| **v0.2** | 闭源 API 诊断：原图 vs 退化 CT 对比 |
| **v0.3** | MCP 工具增强：API + 工具 vs 纯 API |
| **v0.4** | 智能体规划：LLM 驱动的感知-规划-执行 |
| **v0.5** | RL 训练：verl 策略优化 |

---

## 核心设计模式

1. **注册表模式** (`ToolRegistry`): 装饰器注册工具，解耦工具定义与执行器
2. **MCP 风格工具**: 返回结构化 JSON，专为 LLM 智能体交互设计
3. **委托模式**: 领域逻辑（planner/downstream）与 LLM 交互（llm/）分离
4. **回退层级**: JSON解析 → 正则 → 原文回退；LLM规划 → 规则规划回退
5. **组合优于继承**: 流水线由 detector, planner, executor, judge 模块组合而成，依赖注入
