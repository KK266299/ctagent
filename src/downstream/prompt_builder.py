# ============================================================================
# 模块职责: Prompt 构建器 — 为闭源 VLM API 构建 CT 诊断 prompt
#   支持: 直接诊断 prompt / tool-augmented prompt / 对比实验 prompt
# 参考: MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro) — prompt design
#       4KAgent (https://github.com/taco-group/4KAgent) — perception prompt
# ============================================================================
from __future__ import annotations

import json
from typing import Any


# -- 默认 system prompt --
DEFAULT_SYSTEM_PROMPT = """\
You are an expert radiologist AI assistant specialized in CT image analysis.
You provide structured diagnostic assessments based on CT images and any
supplementary analysis data provided.

Always respond in the following JSON format:
{
  "findings": ["<finding1>", "<finding2>", ...],
  "diagnosis": "<primary diagnosis>",
  "confidence": <float 0-1>,
  "severity": "<normal|mild|moderate|severe>",
  "reasoning": "<brief reasoning>"
}
"""

# -- 直接诊断 prompt 模板 --
DIRECT_DIAGNOSIS_TEMPLATE = """\
Please analyze the following CT image and provide a diagnostic assessment.

{context_hint}

Provide your analysis in the structured JSON format as instructed.
"""

# -- Tool-augmented prompt 模板 --
TOOL_AUGMENTED_TEMPLATE = """\
Please analyze the following CT image. In addition to the image itself,
the following automated analysis results are available:

## Automated Tool Analysis Results
{tool_results_text}

Use both the image and the tool analysis results to provide a comprehensive
diagnostic assessment in the structured JSON format as instructed.
"""

# -- 对比 prompt 模板 --
COMPARISON_TEMPLATE = """\
You are comparing diagnostic outcomes between original and degraded CT images.

Image context: {context}

Please provide your diagnostic assessment for this {context} CT image
in the structured JSON format as instructed.
"""


class PromptBuilder:
    """CT 诊断 Prompt 构建器。"""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def build_direct_diagnosis_prompt(self, **kwargs: Any) -> str:
        """模式 1: 直接诊断 prompt。"""
        context = kwargs.get("context", "")
        if context:
            context_hint = f"Image context: {context}"
        else:
            context_hint = ""
        return DIRECT_DIAGNOSIS_TEMPLATE.format(context_hint=context_hint).strip()

    def build_tool_augmented_prompt(
        self,
        tool_results: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """模式 2: Tool-augmented prompt — 附加工具分析结果。"""
        tool_results_text = self._format_tool_results(tool_results)
        return TOOL_AUGMENTED_TEMPLATE.format(tool_results_text=tool_results_text).strip()

    def build_comparison_prompt(self, context: str = "original", **kwargs: Any) -> str:
        """对比实验 prompt。"""
        return COMPARISON_TEMPLATE.format(context=context).strip()

    def build_tool_descriptions(self, tools: dict[str, str]) -> str:
        """构建可用工具描述文本（供 agent planner 使用）。

        Args:
            tools: {tool_name: description} 映射

        Returns:
            格式化的工具描述文本
        """
        lines = []
        for name, desc in tools.items():
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _format_tool_results(self, tool_results: dict[str, Any]) -> str:
        """将工具结果格式化为可读文本。"""
        sections = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict):
                formatted = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                formatted = str(result)
            sections.append(f"### {tool_name}\n```json\n{formatted}\n```")
        return "\n\n".join(sections)
