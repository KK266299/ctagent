# ============================================================================
# 模块职责: verl 框架适配器 — 将 CT 修复环境对接到 verl 的 RL 训练管线
#   当前为占位实现，定义接口和数据转换逻辑
#   后续: 实现与 verl 的 DataProto / Actor / Rollout 对接
# 参考: verl (https://github.com/verl-project/verl) — training pipeline
#       verl DataProto, ActorRolloutRef 等核心接口
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

from src.rl.env import CTRestorationEnv
from src.rl.reward import RewardFunction, RewardConfig
from src.rl.trajectory import Trajectory, Transition, TrajectoryBuffer

logger = logging.getLogger(__name__)


class VerlAdapter:
    """verl 框架适配器（占位实现）。

    负责:
    1. 将 CTRestorationEnv 包装为 verl 兼容的环境
    2. 将 Trajectory 数据转为 verl 的 DataProto 格式
    3. 提供 rollout / training 入口

    Usage (未来):
        adapter = VerlAdapter(env_config={...})
        adapter.collect_rollouts(num_episodes=100)
        adapter.prepare_training_data()
        adapter.train(num_epochs=10)
    """

    def __init__(
        self,
        env_config: dict[str, Any] | None = None,
        reward_config: RewardConfig | None = None,
        buffer_size: int = 10000,
    ) -> None:
        self.env_config = env_config or {}
        self.reward_fn = RewardFunction(reward_config)
        self.buffer = TrajectoryBuffer(max_size=buffer_size)
        self._env: CTRestorationEnv | None = None

    def create_env(self) -> CTRestorationEnv:
        """创建 RL 环境。"""
        self._env = CTRestorationEnv(**self.env_config)
        return self._env

    def collect_rollout(
        self,
        image: Any,
        policy: Any = None,
        reference: Any = None,
    ) -> Trajectory:
        """收集单个 episode 的 rollout 数据。

        Args:
            image: 输入图像
            policy: 策略模型（None 则使用随机策略）
            reference: 参考图像（可选）

        Returns:
            Trajectory 记录
        """
        import numpy as np

        env = self._env or self.create_env()
        state = env.reset(image, reference=reference)
        trajectory = Trajectory()

        while not state.done:
            obs = env.get_observation()
            # 选择 action
            if policy is not None:
                action = self._policy_predict(policy, obs)
            else:
                # 随机策略
                action = np.random.choice(env.action_space)

            obs_before = obs.copy()
            state, reward, done, info = env.step(action)
            obs_after = env.get_observation()

            trajectory.add_transition(Transition(
                step=state.step_count - 1,
                action=action,
                reward=reward,
                done=done,
                observation_before=obs_before,
                observation_after=obs_after,
                info=info,
            ))

        # 计算 episode 总奖励
        trajectory.episode_rewards = self.reward_fn.compute_episode_reward(
            original_image=env._original_image,
            final_image=state.image,
            reference=reference,
            num_steps=trajectory.num_steps,
        )
        self.buffer.add(trajectory)
        return trajectory

    def _policy_predict(self, policy: Any, observation: dict) -> str:
        """调用 policy 模型预测 action（占位）。"""
        # TODO: 对接 verl policy model
        # 期望接口: action_str = policy.predict(observation)
        raise NotImplementedError("Policy prediction not yet implemented")

    # ------------------------------------------------------------------
    # verl DataProto 转换 (占位)
    # ------------------------------------------------------------------

    def trajectories_to_verl_data(self) -> Any:
        """将 trajectory buffer 转为 verl DataProto 格式。

        TODO: 实现与 verl.DataProto 的对接
        verl DataProto 通常包含:
        - obs: observation tensors
        - actions: action tensors
        - rewards: reward tensors
        - dones: done flags
        - log_probs: policy log probabilities
        """
        logger.info(
            "Converting %d trajectories to verl format (placeholder)",
            len(self.buffer),
        )
        # 占位: 返回 dict 格式
        all_data = []
        for traj in self.buffer.buffer:
            all_data.append(traj.to_dict())
        return all_data

    # ------------------------------------------------------------------
    # 训练入口 (占位)
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 1, **kwargs: Any) -> dict[str, Any]:
        """RL 训练入口（占位）。

        TODO: 对接 verl trainer
        - PPOTrainer / GRPOTrainer
        - Actor-Critic 架构
        - 数据并行
        """
        logger.warning("RL training is not yet implemented. This is a placeholder.")
        return {
            "status": "placeholder",
            "num_epochs": num_epochs,
            "buffer_size": len(self.buffer),
            "buffer_stats": self.buffer.statistics(),
        }

    def load_policy(self, checkpoint: str) -> Any:
        """加载训练好的 policy（占位）。

        TODO: 实现 verl checkpoint 加载
        """
        logger.info("Loading policy from %s (placeholder)", checkpoint)
        return None
