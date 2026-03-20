# CT-Agent-Frontdoor

> CT 图像退化感知 → Planner → 多工具修复 → MedAgent-Pro 诊断 → Judge/Eval

一个基于 Agent 架构的 CT 图像质量增强与诊断系统。与 LLaMA-Factory 并列部署，不修改其源码。

## 架构概览

```
输入 CT 图像
    │
    ▼
┌─────────────────────────────────────────────────┐
│            MCP-Style 感知工具层                    │
│  AnalysisTool · PerceptionTool · StatisticsTool  │
│                src/tools/mcp_style/              │
└──────────────────────┬──────────────────────────┘
                       │ 结构化 JSON 分析结果
                       ▼
┌─────────────────────────────────────────────────┐
│                  Planner 层                      │
│  RuleBased ──→ AgentBased (LLM) ──→ RL Policy   │
│                src/planner/                      │
└──────────────────────┬──────────────────────────┘
                       │ Plan (工具调用序列)
                       ▼
┌─────────────────────────────────────────────────┐
│                Executor 执行引擎                  │
│  ┌───────────┐  ┌─────────────┐                  │
│  │ Classical  │  │ CT-Specific │                  │
│  │ NLM/USM/  │  │ MAR/LDCT/SR │                  │
│  │ CLAHE     │  │ PromptCT    │                  │
│  └───────────┘  └─────────────┘                  │
│         src/tools/ + src/executor/               │
└──────────────────────┬──────────────────────────┘
                       │ 修复后图像
                       ▼
┌─────────────────────────────────────────────────┐
│           闭源 VLM 诊断 (Downstream)              │
│                                                  │
│  模式 A: API 直接诊断 (GPT-4o / Claude)           │
│  模式 B: API + MCP Tools 增强诊断                 │
│  模式 C: 原始 vs 退化 对比诊断                     │
│         src/downstream/                          │
└──────────────────────┬──────────────────────────┘
                       │ DiagnosisResult
                       ▼
┌─────────────────────────────────────────────────┐
│              Judge / Eval                        │
│  QualityJudge (IQA 前后对比)                      │
│  PipelineEvaluator (PSNR/SSIM/端到端报告)         │
│         src/judge/ + src/eval/                   │
└─────────────────────────────────────────────────┘
                       │
                       ▼  (预留)
┌─────────────────────────────────────────────────┐
│              RL 训练接口                          │
│  Env · Reward · Trajectory · VerlAdapter         │
│         src/rl/                                  │
└─────────────────────────────────────────────────┘
```

## 三条可运行链路

```python
# 链路 1: 闭源 API 直接诊断 — 对比退化前后表现差异
adapter = ClosedAPIAdapter(config=APIConfig(model="gpt-4o"))
results = adapter.compare_diagnosis(original_image, degraded_image)

# 链路 2: API + MCP Tools 增强诊断
planner = AgentBasedPlanner()
perceptions = planner.collect_perceptions(image)  # analysis + perception + statistics
result = adapter.predict_with_tools(image, tool_results=perceptions)

# 链路 3: Agent 规划 → 执行 → 诊断
plan = planner.plan_with_perception(image)  # LLM 驱动
results = Executor().execute(plan, image)
diagnosis = adapter.predict(results[-1].image)
```

## 参考项目

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — LLM 微调框架
- [MedAgent-Pro](https://github.com/jinlab-imvr/MedAgent-Pro) — 医学 Agent 架构
- [JarvisIR](https://github.com/LYL1015/JarvisIR) — 图像修复 Agent
- [MedQ-Bench](https://github.com/liujiyaoFDU/MedQ-Bench) — 医学质量评估 benchmark
- [CAPIQA](https://github.com/aaz-imran/capiqa) — CT 感知 IQA
- [ProCT](https://github.com/Masaaki-75/proct) — CT 重建
- [PromptCT](https://github.com/shibaoshun/PromptCT) — Prompt 驱动 CT 增强
- [RISE-MAR](https://github.com/Masaaki-75/rise-mar) — 金属伪影去除
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) — 图像质量评估工具包
- [4KAgent](https://github.com/taco-group/4KAgent) — Perception-Planning-Execution Agent
- [Earth-Agent](https://github.com/opendatalab/Earth-Agent) — MCP-style tool design
- [verl](https://github.com/verl-project/verl) — RL 训练框架

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行 pipeline (rule-based)
python scripts/run_pipeline.py --config configs/experiment/minimal.yaml --input /path/to/ct.dcm

# 运行评估
python scripts/evaluate.py --config configs/experiment/minimal.yaml \
  --restored-dir output/restored --reference-dir data/clean

# 运行测试
pytest tests/
```

## 目录结构

```
ct-agent-frontdoor/
├── configs/
│   ├── data/                  # 数据路径与配置
│   ├── degradation/           # 退化模型配置 (noise/artifact/blur/resolution)
│   ├── tools/                 # 工具配置 (classical + ct_specific)
│   ├── planner/               # Planner 配置 (rule_based + llm_based)
│   └── experiment/            # 实验配置 (minimal + full_pipeline)
├── src/
│   ├── io/                    # 数据 I/O (DICOM, NIfTI, PNG) + CT 窗宽窗位
│   ├── datasets/              # CTDataset + PairedCTDataset
│   ├── degradations/          # 退化检测器 + 模拟器 + 类型定义
│   ├── iqa/                   # 全参考 (PSNR/SSIM) + 无参考 IQA
│   ├── tools/
│   │   ├── classical/         # NLM / Gaussian / USM / CLAHE
│   │   ├── ct/                # MAR / LDCT Denoise / Super Resolution
│   │   └── mcp_style/         # MCP-style 工具 (JSON 输出, 供 LLM 调用)
│   │       ├── analysis_tool  #   退化类型分析
│   │       ├── perception_tool#   IQA 指标感知
│   │       ├── restoration_tool#  修复 + 前后对比
│   │       └── statistics_tool#   HU 分布 / ROI 统计
│   ├── planner/
│   │   ├── rule_planner       # 基于规则 (MVP)
│   │   ├── agent_based        # LLM Agent (perception → plan → execute)
│   │   └── policy_rl_placeholder # RL 策略接口 (占位)
│   ├── executor/              # 串行工具链执行引擎
│   ├── downstream/
│   │   ├── closed_api_adapter # GPT-4o / Claude VLM 诊断适配器
│   │   ├── prompt_builder     # CT 诊断 prompt 构建
│   │   ├── response_parser    # 结构化响应解析
│   │   └── classifier         # 本地分类器 (占位)
│   ├── judge/                 # 修复质量评判 (有参考 / 无参考)
│   ├── eval/                  # 端到端评估 + JSON 报告
│   ├── rl/                    # RL 训练接口 (占位, verl 兼容)
│   │   ├── env                #   Gym-style CT 修复环境
│   │   ├── reward             #   多维度奖励函数
│   │   ├── trajectory         #   Episode 记录与缓冲
│   │   └── verl_adapter       #   verl 框架适配器
│   └── utils/                 # 配置加载 / 种子 / 日志
├── scripts/
│   ├── run_pipeline.py        # 端到端 pipeline 入口
│   ├── evaluate.py            # 批量评估
│   └── simulate_degradation.py# 退化模拟
└── tests/                     # 单元测试 (iqa/degradation/tools/planner/io/api/mcp/rl)
```

## 开发路线

1. **v0.1 (当前)**: 最小可行系统 — rule-based planner + classical tools + IQA 评估
2. **v0.2**: 闭源 API 诊断对比 — 原始 vs 退化 CT 诊断差异量化
3. **v0.3**: MCP Tools 增强 — API + tools vs API 直接诊断对比
4. **v0.4**: Agent 规划 — LLM-driven perception-planning-execution
5. **v0.5**: RL 训练 — verl policy 优化

## License

MIT
