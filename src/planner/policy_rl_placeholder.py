# ============================================================================
# 模块职责: RL Policy 占位 — 预留基于强化学习的规划策略接口
#   后续使用 verl 做 policy 优化时，此模块将被替换为真正的 RL policy
#   当前仅定义接口，内部回退到 rule-based 逻辑
# 参考: verl (https://github.com/verl-project/verl) — RL training framework
#       4KAgent (https://github.com/taco-group/4KAgent) — agent policy
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.degradations.types import DegradationReport
from src.planner.base import BasePlanner, Plan, ToolCall

logger = logging.getLogger(__name__)


class RLPolicyPlanner(BasePlanner):
    """基于 RL 策略的 Planner（占位实现）。

    接口设计:
    - state: 图像特征 + 退化信息 + 已执行工具历史
    - action: 选择下一个工具 (或 stop)
    - reward: IQA 提升 + 诊断准确性 (来自 src/rl/reward.py)

    当前实现: 回退到 RuleBasedPlanner。
    后续: 加载 verl 训练好的 policy model 做推理。
    """

    def __init__(
        self,
        policy_checkpoint: str | None = None,
        device: str = "cuda",
        max_steps: int = 3,
    ) -> None:
        self.policy_checkpoint = policy_checkpoint
        self.device = device
        self.max_steps = max_steps
        self._policy_model: Any = None

    def plan(self, report: DegradationReport, **kwargs: Any) -> Plan:
        """RL policy 规划接口。"""
        image = kwargs.get("image")
        if self._policy_model is not None and image is not None:
            return self._rl_plan(image, report)
        # 回退
        logger.info("RL policy not loaded, falling back to rule-based planner")
        from src.planner.rule_planner import RuleBasedPlanner
        return RuleBasedPlanner(max_chain=self.max_steps).plan(report)

    def _rl_plan(self, image: np.ndarray, report: DegradationReport) -> Plan:
        """使用 RL policy 进行规划（占位）。"""
        # TODO: 实现 RL 推理循环
        # state = self._extract_state(image, report)
        # steps = []
        # for _ in range(self.max_steps):
        #     action = self._policy_model.predict(state)
        #     if action == STOP_ACTION:
        #         break
        #     steps.append(ToolCall(tool_name=action.tool_name, params=action.params))
        #     state = self._update_state(state, action)
        raise NotImplementedError("RL policy inference not yet implemented")

    def load_policy(self, checkpoint: str | None = None) -> None:
        """加载 RL policy 权重。"""
        path = checkpoint or self.policy_checkpoint
        if path is None:
            logger.warning("No policy checkpoint specified")
            return
        # TODO: 加载 verl 训练的 policy
        # from src.rl.verl_adapter import VerlAdapter
        # adapter = VerlAdapter()
        # self._policy_model = adapter.load_policy(path)
        logger.info("Policy loading placeholder: %s", path)

    def extract_state(self, image: np.ndarray, report: DegradationReport) -> dict[str, Any]:
        """提取 RL state（供外部使用和 trajectory 记录）。

        State 包含:
        - image_features: 图像统计特征
        - degradation_info: 退化检测结果
        - history: 已执行工具记录
        """
        return {
            "image_shape": list(image.shape),
            "image_mean": float(np.mean(image)),
            "image_std": float(np.std(image)),
            "degradations": [
                {"type": d.value, "severity": s.value}
                for d, s in report.degradations
            ],
            "iqa_scores": report.iqa_scores,
            "history": [],
        }
