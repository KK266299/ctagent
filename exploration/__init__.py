# ============================================================================
# 模块职责: RL / Policy Learning 数据准备接口
#   当前阶段: 从 src/rl/ re-export，建立 exploration/ 顶层入口
#   后续阶段(Phase 2): 将 src/rl/ 代码迁入此处
#   语义说明: "exploration" 而非 "rl"，强调此模块的职责是
#     为后续 policy learning 准备数据（env / reward / trajectory），
#     而不是直接做 RL 训练
# 参考: src/rl/ — 当前实现 (env, reward, trajectory, verl_adapter)
#       verl — RL training framework
#       AgenticIR — exploration module
# ============================================================================

from src.rl.env import CTRestorationEnv
from src.rl.reward import RewardFunction, RewardConfig
from src.rl.trajectory import Trajectory, Transition, TrajectoryBuffer
from src.rl.verl_adapter import VerlAdapter

__all__ = [
    "CTRestorationEnv",
    "RewardFunction",
    "RewardConfig",
    "Trajectory",
    "Transition",
    "TrajectoryBuffer",
    "VerlAdapter",
]
