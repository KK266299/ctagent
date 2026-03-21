# ============================================================================
# 模块职责: 执行引擎 — 按 Plan 串行执行工具链，产出结构化 ExecutionTrace
#   每步记录: tool_name / params / success / elapsed_ms / image_shape / metadata
#   trace 通过 last_trace 属性访问，供 judge / memory / exploration 消费
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — tool execution
#       AgenticIR — executor with trace & rollback
#       4KAgent — execution logging for reflection
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.planner.base import Plan
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry
from executor.trace import ExecutionTrace, TraceRecorder

logger = logging.getLogger(__name__)


class Executor:
    """按 Plan 顺序执行工具链，同时产出 ExecutionTrace。"""

    def __init__(self, tool_configs: dict[str, Any] | None = None) -> None:
        self.tool_configs = tool_configs or {}
        self._tool_cache: dict[str, Any] = {}
        self._last_trace: ExecutionTrace | None = None

    def execute(self, plan: Plan, image: np.ndarray) -> list[ToolResult]:
        """执行计划中的所有工具。

        Args:
            plan: 工具调用计划
            image: 输入图像

        Returns:
            每步的 ToolResult 列表 (返回值类型不变，向后兼容)

        Side effect:
            self.last_trace 被更新为本次执行的 ExecutionTrace
        """
        recorder = TraceRecorder.begin(
            plan_reasoning=plan.reasoning,
            plan_tool_names=plan.tool_names(),
        )

        results: list[ToolResult] = []
        current_image = image

        for i, step in enumerate(plan.steps):
            logger.info("Executing tool: %s", step.tool_name)
            recorder.step_start()

            try:
                tool = self._get_tool(step.tool_name)
                result = tool.run(current_image, **step.params)
                results.append(result)
                recorder.record_step(
                    step_index=i,
                    tool_name=step.tool_name,
                    params=step.params,
                    success=result.success,
                    message=result.message,
                    image_shape=tuple(result.image.shape),
                    metadata=result.metadata,
                )
                if result.success:
                    current_image = result.image
                else:
                    logger.warning("Tool %s failed: %s", step.tool_name, result.message)

            except Exception as e:
                logger.error("Tool %s raised exception: %s", step.tool_name, e)
                recorder.record_step(
                    step_index=i,
                    tool_name=step.tool_name,
                    params=step.params,
                    success=False,
                    message=str(e),
                    image_shape=tuple(current_image.shape),
                )
                results.append(ToolResult(
                    image=current_image,
                    tool_name=step.tool_name,
                    success=False,
                    message=str(e),
                ))

        self._last_trace = recorder.finish()
        logger.info(
            "Execution complete: %d steps, %d success, %d failed, %.1fms",
            len(results),
            self._last_trace.num_success,
            self._last_trace.num_failed,
            self._last_trace.total_elapsed_ms,
        )
        return results

    @property
    def last_trace(self) -> ExecutionTrace | None:
        """最近一次 execute() 产出的 trace。"""
        return self._last_trace

    def _get_tool(self, name: str) -> Any:
        """获取或创建工具实例。"""
        if name not in self._tool_cache:
            kwargs = self.tool_configs.get(name, {})
            self._tool_cache[name] = ToolRegistry.create(name, **kwargs)
        return self._tool_cache[name]
