# ============================================================================
# 模块职责: 执行引擎 — 串行执行 Plan 中的工具调用
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — tool execution
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.planner.base import Plan
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Executor:
    """按 Plan 顺序执行工具链。"""

    def __init__(self, tool_configs: dict[str, Any] | None = None) -> None:
        self.tool_configs = tool_configs or {}
        self._tool_cache: dict[str, Any] = {}

    def execute(self, plan: Plan, image: np.ndarray) -> list[ToolResult]:
        """执行计划中的所有工具。

        Args:
            plan: 工具调用计划
            image: 输入图像

        Returns:
            每步的 ToolResult 列表
        """
        results: list[ToolResult] = []
        current_image = image

        for step in plan.steps:
            logger.info("Executing tool: %s", step.tool_name)
            try:
                tool = self._get_tool(step.tool_name)
                result = tool.run(current_image, **step.params)
                results.append(result)
                if result.success:
                    current_image = result.image
                else:
                    logger.warning("Tool %s failed: %s", step.tool_name, result.message)
            except Exception as e:
                logger.error("Tool %s raised exception: %s", step.tool_name, e)
                results.append(ToolResult(
                    image=current_image,
                    tool_name=step.tool_name,
                    success=False,
                    message=str(e),
                ))

        return results

    def _get_tool(self, name: str) -> Any:
        """获取或创建工具实例。"""
        if name not in self._tool_cache:
            kwargs = self.tool_configs.get(name, {})
            self._tool_cache[name] = ToolRegistry.create(name, **kwargs)
        return self._tool_cache[name]

    @property
    def final_image(self) -> np.ndarray | None:
        """便捷属性：获取最近一次执行的最终图像。"""
        # 由 execute() 的调用者从 results 中获取
        return None
