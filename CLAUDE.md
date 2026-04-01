# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run single-image pipeline (rule-based)
python scripts/run_pipeline.py --config configs/experiment/minimal.yaml --input /path/to/ct.dcm

# Batch evaluation
python scripts/evaluate.py --config configs/experiment/minimal.yaml \
  --restored-dir output/restored --reference-dir data/clean

# CQ500 evaluations
python scripts/run_cq500_iqa_eval.py   # IQA (PSNR/SSIM)
python scripts/run_cq500_api_eval.py   # Diagnosis accuracy
python scripts/run_formal_comparison.py  # Multi-strategy comparison

# Train DnCNN
python scripts/train_dncnn.py

# Tests
pytest tests/
pytest tests/test_planner.py  # single module

# Lint
ruff check src/ tests/
```

## Architecture

The system is a perception-planning-execution agent for CT image restoration:

```
Input CT → DegradationDetector → Planner → Executor (tool chain) → [Downstream VLM] → Judge/Eval
```

**Key layers:**

- **`src/io/`** — CT I/O (DICOM, NIfTI, PNG, NumPy), HU windowing. Entry point for all image loading.
- **`src/degradations/`** — `DegradationDetector` outputs a `DegradationReport` with 11 degradation types + severity. `simulator.py` synthesizes degraded images for experiments.
- **`src/planner/`** — Three planners sharing a common interface:
  - `RuleBasedPlanner` (rule_planner.py): hardcoded (degradation, severity) → tool list. Production baseline.
  - `AgentBasedPlanner` (agent_based.py): calls MCP-style perception tools, then delegates to `llm/planner_caller.py` for LLM-driven plan generation. Falls back to rule-based.
  - `PolicyRLPlaceholder`: skeleton for future RL policy.
- **`src/tools/`** — Tool implementations + `registry.py` (centralized instantiation with caching). Subcategories:
  - `classical/`: NLM, BM3D, Median, Wavelet, USM, CLAHE, MAR, Ring Removal, etc.
  - `ct/`: LDCT denoise, MAR, super-resolution adapters.
  - `learned/`: DnCNN (active); SR adapter (placeholder).
  - `mcp_style/`: JSON-outputting tools for LLM consumption — `AnalysisTool`, `PerceptionTool`, `StatisticsTool`, `RestorationTool`.
- **`src/executor/`** — `engine.py` executes tool sequences, records `ExecutionTrace` per step.
- **`src/downstream/`** — `ClosedAPIAdapter` bridges to GPT-4o/Claude for VLM diagnosis; `prompt_builder.py` + `response_parser.py`.
- **`src/judge/`** — `QualityJudge`: with-reference (PSNR/SSIM) and no-reference composite scoring.
- **`src/iqa/`** — IQA metrics (full-reference and no-reference via pyiqa).
- **`src/rl/`** — RL environment, reward, trajectory, verl adapter (all placeholder/skeleton).
- **`llm/`** — API client wrappers, `planner_caller.py`, `diagnosis_caller.py`, prompt/response utilities.
- **`pipeline/`** — Orchestration: `single_pass.py` (one-shot), `agent_loop.py` (iterative Plan→Execute→Judge→Replan), `api_guided_planner.py`.
- **`dataset/`** — DICOM scanning, `CTDataset`/`PairedCTDataset`, MAR synthesis pipeline (`dataset/mar/`).
- **`eval/`** — Experiment-level evaluation for CQ500 and BHX datasets.

**Configuration** is YAML-driven under `configs/`. Start from `configs/experiment/minimal.yaml` for the MVP. Planner strategy is set in `configs/planner/`.

**Data flow detail:**

```python
# Rule-based (minimal)
report = DegradationDetector().detect(image)      # → DegradationReport
plan   = RuleBasedPlanner().plan(report)           # → Plan (ordered ToolCall list)
results = Executor().execute(plan, image)          # → list[ToolResult] with ExecutionTrace

# LLM-enhanced
perceptions = AgentBasedPlanner().collect_perceptions(image)  # calls mcp_style tools
plan = planner_caller.call(**perceptions)                      # LLM → Plan
```

## Notes

- The project is deployed alongside LLaMA-Factory; do not modify LLaMA-Factory source.
- `src/rl/` and `policy_rl_placeholder.py` are stubs; the RL training path (verl) is not yet active.
- `dataset/mar/` contains a dedicated MAR synthesis pipeline documented in `docs/mar_degradation_pipeline.md`.
- Ruff line length is 120, target Python 3.9+.
