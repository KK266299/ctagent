# ============================================================================
# 模块职责: 执行 Trace — 记录 executor 每步执行的完整信息
#   用途:
#   1. 闭环 pipeline 中 judge 的评判依据
#   2. memory 模块记录经验的数据来源
#   3. exploration 模块生成 trajectory 的数据来源
#   4. 调试与可视化
# 参考: 4KAgent — execution log / reflection data
#       AgenticIR — executor trace for rollback
#       verl — trajectory step recording
# ============================================================================
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TraceStep:
    """单步执行记录。

    Attributes:
        step_index: 在 Plan 中的执行顺序 (0-based)
        tool_name: 工具名称
        params: 工具参数
        success: 是否成功
        message: 执行消息 (错误信息 or 成功描述)
        elapsed_ms: 执行耗时 (毫秒)
        image_shape: 输出图像的 shape (不存图像本身)
        image_hash: 输出图像的快速 hash (用于去重/比对)
        metadata: 工具返回的额外元信息
    """
    step_index: int
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    message: str = ""
    elapsed_ms: float = 0.0
    image_shape: tuple[int, ...] = ()
    image_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["image_shape"] = list(self.image_shape)
        return d


@dataclass
class ExecutionTrace:
    """一次完整 Plan 执行的 trace 记录。

    Attributes:
        trace_id: 唯一标识
        plan_reasoning: 所执行 Plan 的 reasoning 字段
        plan_tool_names: 所执行 Plan 的工具名序列
        steps: 每步 TraceStep 列表
        start_time: 开始时间 (ISO format)
        end_time: 结束时间 (ISO format)
        total_elapsed_ms: 总耗时
        num_success: 成功步数
        num_failed: 失败步数
        metadata: 扩展字段
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    plan_reasoning: str = ""
    plan_tool_names: list[str] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_elapsed_ms: float = 0.0
    num_success: int = 0
    num_failed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep) -> None:
        """添加一步执行记录并更新统计。"""
        self.steps.append(step)
        if step.success:
            self.num_success += 1
        else:
            self.num_failed += 1
        self.total_elapsed_ms += step.elapsed_ms

    @property
    def all_success(self) -> bool:
        return self.num_failed == 0 and self.num_success > 0

    @property
    def tool_sequence(self) -> list[str]:
        """实际执行的工具序列 (仅成功的)。"""
        return [s.tool_name for s in self.steps if s.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "plan_reasoning": self.plan_reasoning,
            "plan_tool_names": self.plan_tool_names,
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_elapsed_ms": self.total_elapsed_ms,
            "num_success": self.num_success,
            "num_failed": self.num_failed,
            "metadata": self.metadata,
        }


class TraceRecorder:
    """Trace 记录辅助类 — 在 executor 执行过程中构建 ExecutionTrace。

    Usage:
        recorder = TraceRecorder.begin(plan)
        recorder.record_step(step_index=0, tool_name="denoise_nlm", ...)
        trace = recorder.finish()
    """

    def __init__(self, trace: ExecutionTrace) -> None:
        self._trace = trace
        self._step_start: float | None = None

    @classmethod
    def begin(cls, plan_reasoning: str = "", plan_tool_names: list[str] | None = None) -> TraceRecorder:
        """开始一次新的 trace 记录。"""
        trace = ExecutionTrace(
            plan_reasoning=plan_reasoning,
            plan_tool_names=plan_tool_names or [],
            start_time=_now_iso(),
        )
        recorder = cls(trace)
        return recorder

    def step_start(self) -> None:
        """标记一步执行的开始时间。"""
        self._step_start = time.perf_counter()

    def record_step(
        self,
        step_index: int,
        tool_name: str,
        params: dict[str, Any] | None = None,
        success: bool = True,
        message: str = "",
        image_shape: tuple[int, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> TraceStep:
        """记录一步执行结果。"""
        elapsed = 0.0
        if self._step_start is not None:
            elapsed = (time.perf_counter() - self._step_start) * 1000
            self._step_start = None

        step = TraceStep(
            step_index=step_index,
            tool_name=tool_name,
            params=params or {},
            success=success,
            message=message,
            elapsed_ms=round(elapsed, 2),
            image_shape=image_shape,
            metadata=metadata or {},
        )
        self._trace.add_step(step)
        return step

    def finish(self) -> ExecutionTrace:
        """结束 trace 记录，返回完整 ExecutionTrace。"""
        self._trace.end_time = _now_iso()
        return self._trace

    @property
    def trace(self) -> ExecutionTrace:
        return self._trace


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
