# ============================================================================
# 模块职责: MCP-style 工具包 — 可被 LLM Agent 调用的结构化工具
#   这些工具返回 JSON 结构化结果（而非图像），供 VLM 做增强诊断
# 参考: Earth-Agent (https://github.com/opendatalab/Earth-Agent/tree/main/agent/tools)
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — tool design
#       4KAgent (https://github.com/taco-group/4KAgent) — perception tools
# ============================================================================

from src.tools.mcp_style.analysis_tool import AnalysisTool
from src.tools.mcp_style.perception_tool import PerceptionTool
from src.tools.mcp_style.restoration_tool import RestorationTool
from src.tools.mcp_style.statistics_tool import StatisticsTool

ALL_MCP_TOOLS = [AnalysisTool, PerceptionTool, RestorationTool, StatisticsTool]

__all__ = [
    "AnalysisTool",
    "PerceptionTool",
    "RestorationTool",
    "StatisticsTool",
    "ALL_MCP_TOOLS",
]
