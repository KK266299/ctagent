# ============================================================================
# 模块职责: ExperienceRecord — 单次 pipeline 执行的完整经验记录
#   一条 experience = 一次从输入到判定的完整闭环
#   包含: 退化信息、工具序列、judge 结果、downstream 结果、时间戳
#   用途: 为 planner 提供 few-shot 经验 / 为 exploration 提供离线数据
# 参考: AgenticIR — experience buffer
#       4KAgent — episode record for reflection
#       verl — trajectory metadata
# ============================================================================
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ExperienceRecord:
    """单次 pipeline 执行的经验记录。

    Attributes:
        record_id:          唯一标识
        timestamp:          记录时间 (ISO format)
        degradation_types:  检测到的退化类型列表 (e.g. ["noise", "blur"])
        degradation_report: 完整退化检测报告
        tool_sequence:      实际执行的工具序列 (e.g. ["denoise_nlm", "sharpen_usm"])
        plan_reasoning:     planner 的推理过程
        quality_verdict:    quality judge 的判定结果
        safety_verdict:     safety judge 的判定结果
        downstream_result:  下游诊断结果 (prediction, confidence, ...)
        iqa_before:         修复前 IQA 指标
        iqa_after:          修复后 IQA 指标
        success:            此经验是否为"成功"路径
        iteration_count:    闭环迭代了几轮
        metadata:           扩展字段
    """
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    degradation_types: list[str] = field(default_factory=list)
    degradation_report: dict[str, Any] = field(default_factory=dict)
    tool_sequence: list[str] = field(default_factory=list)
    plan_reasoning: str = ""
    quality_verdict: dict[str, Any] = field(default_factory=dict)
    safety_verdict: dict[str, Any] = field(default_factory=dict)
    downstream_result: dict[str, Any] = field(default_factory=dict)
    iqa_before: dict[str, float] = field(default_factory=dict)
    iqa_after: dict[str, float] = field(default_factory=dict)
    success: bool = False
    iteration_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperienceRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def route_key(self) -> str:
        """经验路由 key: 退化类型组合 → 工具序列。用于快速匹配。"""
        deg = "+".join(sorted(self.degradation_types)) if self.degradation_types else "none"
        tools = "->".join(self.tool_sequence) if self.tool_sequence else "none"
        return f"{deg}|{tools}"
