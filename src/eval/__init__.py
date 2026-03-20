# ============================================================================
# 模块职责: 评估模块 — 端到端 pipeline 评估与报告生成
# 参考: MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench) — benchmark eval
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch) — metric suite
# ============================================================================

from src.eval.evaluator import PipelineEvaluator

__all__ = ["PipelineEvaluator"]
