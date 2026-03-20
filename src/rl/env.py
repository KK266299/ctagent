# ============================================================================
# 模块职责: CT 修复 RL 环境 — Gym-style 接口
#   State: 当前图像特征 + 退化信息 + 已执行历史
#   Action: 选择一个工具 (或 stop)
#   Reward: IQA 提升 + 诊断准确性
#   Done: 达到最大步数 or agent 选择 stop
# 参考: verl (https://github.com/verl-project/verl) — env interface
#       4KAgent (https://github.com/taco-group/4KAgent) — restoration env
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.degradations.detector import DegradationDetector
from src.iqa.no_reference import NoReferenceIQA
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# 特殊 action: 停止修复
STOP_ACTION = "__stop__"


@dataclass
class EnvState:
    """环境状态。"""
    image: np.ndarray
    step_count: int
    history: list[str]  # 已执行工具名列表
    iqa_scores: dict[str, float]
    done: bool = False


class CTRestorationEnv:
    """CT 图像修复 RL 环境（占位实现）。

    遵循 Gym-style 接口:
        state = env.reset(image)
        state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        max_steps: int = 3,
        available_tools: list[str] | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.available_tools = available_tools or ToolRegistry.list_tools()
        self.action_space = self.available_tools + [STOP_ACTION]

        self._detector = DegradationDetector()
        self._nr_iqa = NoReferenceIQA()
        self._state: EnvState | None = None
        self._original_image: np.ndarray | None = None
        self._reference_image: np.ndarray | None = None

    @property
    def num_actions(self) -> int:
        return len(self.action_space)

    def reset(
        self,
        image: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> EnvState:
        """重置环境。"""
        self._original_image = image.copy()
        self._reference_image = reference
        iqa = self._nr_iqa.evaluate(image)
        self._state = EnvState(
            image=image.copy(),
            step_count=0,
            history=[],
            iqa_scores=iqa,
        )
        return self._state

    def step(self, action: str) -> tuple[EnvState, float, bool, dict[str, Any]]:
        """执行一步。

        Args:
            action: 工具名 或 STOP_ACTION

        Returns:
            (new_state, reward, done, info)
        """
        assert self._state is not None, "Call reset() first"
        info: dict[str, Any] = {"action": action}

        if action == STOP_ACTION or self._state.step_count >= self.max_steps:
            self._state.done = True
            return self._state, 0.0, True, info

        # 执行工具
        try:
            tool = ToolRegistry.create(action)
            result = tool.run(self._state.image)
            new_image = result.image
            info["tool_success"] = result.success
            info["tool_message"] = result.message
        except Exception as e:
            logger.warning("Tool %s failed: %s", action, e)
            new_image = self._state.image
            info["tool_success"] = False
            info["tool_message"] = str(e)

        # 更新 IQA
        new_iqa = self._nr_iqa.evaluate(new_image)
        old_iqa = self._state.iqa_scores

        # 计算 reward (简单版: sharpness 提升)
        reward = new_iqa.get("sharpness", 0) - old_iqa.get("sharpness", 0)
        info["reward_detail"] = {"iqa_before": old_iqa, "iqa_after": new_iqa}

        # 更新状态
        self._state = EnvState(
            image=new_image,
            step_count=self._state.step_count + 1,
            history=self._state.history + [action],
            iqa_scores=new_iqa,
            done=self._state.step_count + 1 >= self.max_steps,
        )
        return self._state, reward, self._state.done, info

    def get_observation(self) -> dict[str, Any]:
        """获取当前 observation 用于 policy 输入。"""
        assert self._state is not None
        return {
            "image_mean": float(np.mean(self._state.image)),
            "image_std": float(np.std(self._state.image)),
            "iqa_scores": self._state.iqa_scores,
            "step_count": self._state.step_count,
            "history": self._state.history,
        }
