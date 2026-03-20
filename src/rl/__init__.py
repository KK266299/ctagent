# ============================================================================
# 模块职责: RL 模块 — 强化学习训练接口（占位）
#   包含: 环境定义 / 奖励函数 / Trajectory 数据 / verl 适配器
#   当前版本仅定义接口，不实现训练逻辑
# 参考: verl (https://github.com/verl-project/verl) — RL training framework
#       4KAgent (https://github.com/taco-group/4KAgent) — agent RL
# ============================================================================

from src.rl.env import CTRestorationEnv
from src.rl.reward import RewardFunction
from src.rl.trajectory import Trajectory, Transition

__all__ = ["CTRestorationEnv", "RewardFunction", "Trajectory", "Transition"]
