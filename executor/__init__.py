# ============================================================================
# 模块职责: 工具执行引擎 — 按 Plan 执行工具链 + trace + rollback
#   trace 模块在此直接暴露; Executor 仍从 src.executor 导入
#   (避免 src.executor.engine ↔ executor 的循环导入)
# 参考: src/executor/engine.py — Executor 实现
#       executor/trace.py — ExecutionTrace / TraceRecorder
# ============================================================================

from executor.trace import ExecutionTrace, TraceStep, TraceRecorder

__all__ = [
    "ExecutionTrace",
    "TraceStep",
    "TraceRecorder",
]


def __getattr__(name: str):
    """延迟导入 Executor，避免循环 import。"""
    if name == "Executor":
        from src.executor.engine import Executor
        return Executor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
