# CT-Agent-Frontdoor

> CT 图像退化感知 → Planner → 多工具修复 → MedAgent-Pro 诊断 → Judge/Eval

一个基于 Agent 架构的 CT 图像质量增强与诊断系统。与 LLaMA-Factory 并列部署，不修改其源码。

## 架构概览

```
输入 CT 图像
    │
    ▼
┌──────────────┐
│  退化感知模块  │  src/degradations/ + src/iqa/
│  (Degradation │
│   Awareness)  │
└──────┬───────┘
       │ 退化类型 + 严重度
       ▼
┌──────────────┐
│   Planner    │  src/planner/
│  (调度规划)   │
└──────┬───────┘
       │ 工具调用序列
       ▼
┌──────────────┐
│   Executor   │  src/executor/
│  (工具执行)   │
│  ┌─────────┐ │
│  │Classical │ │  src/tools/classical/
│  │CT-Specific│  src/tools/ct/
│  └─────────┘ │
└──────┬───────┘
       │ 修复后图像
       ▼
┌──────────────┐
│  Downstream  │  src/downstream/
│  (诊断任务)   │  MedAgent-Pro style
└──────┬───────┘
       │ 诊断结果
       ▼
┌──────────────┐
│  Judge/Eval  │  src/judge/ + src/eval/
│  (质量评估)   │
└──────────────┘
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

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行 pipeline
python scripts/run_pipeline.py --config configs/experiment/minimal.yaml

# 运行测试
pytest tests/
```

## 目录结构

```
ct-agent-frontdoor/
├── configs/
│   ├── data/           # 数据路径与配置
│   ├── degradation/    # 退化模型配置
│   ├── tools/          # 工具配置
│   ├── planner/        # Planner 配置
│   └── experiment/     # 实验配置
├── src/
│   ├── io/             # 数据 I/O (DICOM, NIfTI, PNG)
│   ├── datasets/       # Dataset 定义
│   ├── degradations/   # 退化建模与感知
│   ├── iqa/            # 图像质量评估
│   ├── tools/          # 修复工具 (classical + CT-specific)
│   ├── planner/        # Agent planner
│   ├── executor/       # 工具执行引擎
│   ├── downstream/     # 下游诊断任务
│   ├── judge/          # 修复质量 judge
│   ├── eval/           # 端到端评估
│   └── utils/          # 通用工具
├── scripts/            # 运行脚本
└── tests/              # 测试
```

## License

MIT
