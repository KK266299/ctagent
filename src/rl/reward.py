# ============================================================================
# 模块职责: 奖励函数 — RL 训练用的多维度奖励设计
#   维度: IQA 提升 + 诊断一致性 + 工具效率惩罚
# 参考: verl (https://github.com/verl-project/verl) — reward design
#       4KAgent (https://github.com/taco-group/4KAgent) — quality reward
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.iqa.metrics import compute_metrics
from src.iqa.no_reference import NoReferenceIQA


@dataclass
class RewardConfig:
    """奖励函数配置。"""
    # IQA 改善权重
    iqa_weight: float = 1.0
    # 诊断一致性权重 (与 ground truth 比较)
    diagnosis_weight: float = 0.5
    # 步数惩罚 (每步)
    step_penalty: float = -0.1
    # 失败惩罚 (工具执行失败)
    failure_penalty: float = -0.5
    # 使用的 IQA 指标
    iqa_metric: str = "sharpness"


class RewardFunction:
    """多维度奖励函数。"""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._nr_iqa = NoReferenceIQA()

    def compute(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        reference: np.ndarray | None = None,
        tool_success: bool = True,
        **kwargs: Any,
    ) -> float:
        """计算单步奖励。"""
        reward = 0.0

        # 1. IQA 改善奖励
        iqa_reward = self._iqa_improvement_reward(image_before, image_after, reference)
        reward += self.config.iqa_weight * iqa_reward

        # 2. 步数惩罚
        reward += self.config.step_penalty

        # 3. 失败惩罚
        if not tool_success:
            reward += self.config.failure_penalty

        # 4. 诊断一致性 (需要外部传入)
        diagnosis_reward = kwargs.get("diagnosis_reward", 0.0)
        reward += self.config.diagnosis_weight * diagnosis_reward

        return reward

    def compute_episode_reward(
        self,
        original_image: np.ndarray,
        final_image: np.ndarray,
        reference: np.ndarray | None = None,
        num_steps: int = 0,
        **kwargs: Any,
    ) -> dict[str, float]:
        """计算 episode 总体奖励（用于 trajectory 记录）。"""
        rewards: dict[str, float] = {}

        # IQA 总提升
        iqa_before = self._nr_iqa.evaluate(original_image)
        iqa_after = self._nr_iqa.evaluate(final_image)
        rewards["iqa_improvement"] = (
            iqa_after.get("sharpness", 0) - iqa_before.get("sharpness", 0)
        )

        # 有参考时的指标
        if reference is not None:
            metrics = compute_metrics(final_image, reference, ["psnr", "ssim"])
            rewards["psnr"] = metrics.get("psnr", 0.0)
            rewards["ssim"] = metrics.get("ssim", 0.0)

        # 效率
        rewards["num_steps"] = float(num_steps)
        rewards["step_penalty"] = self.config.step_penalty * num_steps

        # 总奖励
        rewards["total"] = (
            self.config.iqa_weight * rewards["iqa_improvement"]
            + rewards["step_penalty"]
        )
        return rewards

    def _iqa_improvement_reward(
        self,
        before: np.ndarray,
        after: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> float:
        """计算 IQA 改善奖励。"""
        if reference is not None:
            # 有参考: 用 PSNR 变化
            psnr_before = compute_metrics(before, reference, ["psnr"])["psnr"]
            psnr_after = compute_metrics(after, reference, ["psnr"])["psnr"]
            return psnr_after - psnr_before

        # 无参考: 用 sharpness 变化
        iqa_before = self._nr_iqa.evaluate(before)
        iqa_after = self._nr_iqa.evaluate(after)
        return iqa_after.get("sharpness", 0) - iqa_before.get("sharpness", 0)
