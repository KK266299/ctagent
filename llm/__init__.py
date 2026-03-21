# ============================================================================
# 模块职责: 统一 LLM 交互层 — 所有与闭源/开源 LLM 的交互汇聚于此
#   包含: API client 管理、prompt 构建、response 解析、planning/diagnosis caller
#   设计原则: 其他模块(pipeline/downstream/judge)不直接调用 LLM API，全部经由此层
# 参考: AgenticIR — llm module (unified LLM interface)
#       4KAgent — perception & planning prompt design
#       LLaMA-Factory — model/API abstraction layer
# ============================================================================

from llm.api_client import BaseLLMClient, LLMResponse
from llm.planner_caller import PlannerCaller
from llm.diagnosis_caller import DiagnosisCaller

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "PlannerCaller",
    "DiagnosisCaller",
]
