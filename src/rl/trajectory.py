# ============================================================================
# 模块职责: Trajectory 数据结构 — 记录 RL 训练的 episode 数据
#   用于: 离线 RL 数据收集、经验回放、verl 训练数据
# 参考: verl (https://github.com/verl-project/verl) — data format
#       4KAgent (https://github.com/taco-group/4KAgent)
# ============================================================================
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Transition:
    """单步转移。"""
    step: int
    action: str                             # 工具名 或 "__stop__"
    reward: float = 0.0
    done: bool = False
    observation_before: dict[str, Any] = field(default_factory=dict)
    observation_after: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """一条完整的 episode 记录。"""
    episode_id: str = ""
    sample_id: str = ""                     # 来源图像标识
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    episode_rewards: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.transitions)

    @property
    def action_sequence(self) -> list[str]:
        return [t.action for t in self.transitions]

    def add_transition(self, transition: Transition) -> None:
        """添加一步转移。"""
        self.transitions.append(transition)
        self.total_reward += transition.reward

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """保存为 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        """从 JSON 加载。"""
        with open(path) as f:
            data = json.load(f)
        transitions = [Transition(**t) for t in data.pop("transitions", [])]
        return cls(transitions=transitions, **data)


class TrajectoryBuffer:
    """Trajectory 缓冲区 — 收集多条 episode 数据。"""

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.buffer: list[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        """添加一条 trajectory。"""
        self.buffer.append(trajectory)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def __len__(self) -> int:
        return len(self.buffer)

    def save_all(self, output_dir: str | Path) -> None:
        """批量保存所有 trajectory。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, traj in enumerate(self.buffer):
            traj.save(output_dir / f"trajectory_{i:06d}.json")

    def statistics(self) -> dict[str, float]:
        """缓冲区统计。"""
        import numpy as np
        if not self.buffer:
            return {}
        rewards = [t.total_reward for t in self.buffer]
        steps = [t.num_steps for t in self.buffer]
        return {
            "num_episodes": len(self.buffer),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
        }
