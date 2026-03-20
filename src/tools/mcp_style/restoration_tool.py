# ============================================================================
# 模块职责: 修复工具 (MCP 封装) — 调用已有工具链做图像修复，返回修复结果摘要
#   与 src/tools/ 下的 BaseTool 不同：此工具面向 LLM Agent 调用，
#   返回结构化 JSON (含修复前后 IQA 对比)，而非仅返回图像
# 参考: Earth-Agent (https://github.com/opendatalab/Earth-Agent/tree/main/agent/tools)
#       JarvisIR (https://github.com/LYL1015/JarvisIR) — tool orchestration
# ============================================================================
from __future__ import annotations

from typing import Any

import numpy as np

from src.tools.registry import ToolRegistry
from src.iqa.no_reference import NoReferenceIQA


class RestorationTool:
    """MCP-style 修复工具 — 调用底层修复工具并返回结构化报告。

    这是 src/tools/ 中具体修复工具的 MCP 封装层。
    """

    name = "ct_image_restoration"
    description = (
        "Apply a specified restoration tool to a CT image. "
        "Returns the restored image along with before/after quality comparison. "
        "Available tools: denoise_nlm, denoise_gaussian, sharpen_usm, "
        "histogram_clahe, ldct_denoiser, mar_rise, sr_ct."
    )

    parameters_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image identifier"},
            "tool_name": {"type": "string", "description": "Name of restoration tool to apply"},
            "params": {"type": "object", "description": "Tool-specific parameters"},
        },
        "required": ["image", "tool_name"],
    }

    def __init__(self) -> None:
        self.nr_iqa = NoReferenceIQA()

    def __call__(
        self,
        image: np.ndarray,
        tool_name: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """执行修复并返回结构化报告。"""
        params = params or {}

        # 修复前指标
        before_iqa = self.nr_iqa.evaluate(image)

        # 调用底层工具
        try:
            tool = ToolRegistry.create(tool_name)
            tool_result = tool.run(image, **params)
        except Exception as e:
            return {
                "tool": self.name,
                "success": False,
                "error": str(e),
                "applied_tool": tool_name,
            }

        # 修复后指标
        after_iqa = self.nr_iqa.evaluate(tool_result.image)

        return {
            "tool": self.name,
            "success": tool_result.success,
            "applied_tool": tool_name,
            "message": tool_result.message,
            "quality_before": before_iqa,
            "quality_after": after_iqa,
            "improvement": {
                k: after_iqa[k] - before_iqa[k] for k in before_iqa if k in after_iqa
            },
            # 修复后图像存在 metadata 中，不序列化到 JSON
            "_restored_image": tool_result.image,
        }

    def list_available_tools(self) -> list[str]:
        """列出所有可用修复工具。"""
        return ToolRegistry.list_tools()
