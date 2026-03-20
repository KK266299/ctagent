# ============================================================================
# 模块职责: RL 模块单元测试
# ============================================================================
import numpy as np

from src.rl.env import CTRestorationEnv, STOP_ACTION
from src.rl.reward import RewardFunction, RewardConfig
from src.rl.trajectory import Trajectory, Transition, TrajectoryBuffer


def _make_test_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((64, 64)).astype(np.float32)


class TestCTRestorationEnv:
    def test_reset(self):
        # 确保工具已注册
        import src.tools.classical  # noqa: F401
        env = CTRestorationEnv(max_steps=3)
        state = env.reset(_make_test_image())
        assert state.step_count == 0
        assert not state.done
        assert len(state.history) == 0

    def test_step(self):
        import src.tools.classical  # noqa: F401
        env = CTRestorationEnv(max_steps=3)
        env.reset(_make_test_image())
        state, reward, done, info = env.step("denoise_gaussian")
        assert state.step_count == 1
        assert "denoise_gaussian" in state.history
        assert isinstance(reward, float)

    def test_stop_action(self):
        import src.tools.classical  # noqa: F401
        env = CTRestorationEnv(max_steps=3)
        env.reset(_make_test_image())
        state, reward, done, info = env.step(STOP_ACTION)
        assert done
        assert state.done

    def test_max_steps(self):
        import src.tools.classical  # noqa: F401
        env = CTRestorationEnv(max_steps=2)
        env.reset(_make_test_image())
        env.step("denoise_gaussian")
        state, _, done, _ = env.step("denoise_gaussian")
        assert done

    def test_observation(self):
        import src.tools.classical  # noqa: F401
        env = CTRestorationEnv(max_steps=3)
        env.reset(_make_test_image())
        obs = env.get_observation()
        assert "image_mean" in obs
        assert "iqa_scores" in obs


class TestRewardFunction:
    def test_basic_reward(self):
        rf = RewardFunction()
        img = _make_test_image()
        reward = rf.compute(img, img)
        # 同一图像改善应为 0，加上步数惩罚
        assert reward <= 0  # step penalty

    def test_episode_reward(self):
        rf = RewardFunction()
        img = _make_test_image()
        rewards = rf.compute_episode_reward(img, img, num_steps=2)
        assert "total" in rewards
        assert "iqa_improvement" in rewards
        assert "num_steps" in rewards

    def test_with_reference(self):
        rf = RewardFunction()
        img = _make_test_image()
        ref = img.copy()
        rewards = rf.compute_episode_reward(img, img, reference=ref, num_steps=1)
        assert "psnr" in rewards
        assert "ssim" in rewards


class TestTrajectory:
    def test_add_transition(self):
        traj = Trajectory(episode_id="test_001")
        traj.add_transition(Transition(step=0, action="denoise_nlm", reward=0.5))
        traj.add_transition(Transition(step=1, action="sharpen_usm", reward=0.3))
        assert traj.num_steps == 2
        assert traj.total_reward == 0.8
        assert traj.action_sequence == ["denoise_nlm", "sharpen_usm"]

    def test_serialization(self, tmp_path):
        traj = Trajectory(episode_id="test_002")
        traj.add_transition(Transition(step=0, action="denoise_nlm", reward=1.0))
        path = tmp_path / "traj.json"
        traj.save(path)
        loaded = Trajectory.load(path)
        assert loaded.episode_id == "test_002"
        assert loaded.num_steps == 1

    def test_buffer_stats(self):
        buf = TrajectoryBuffer(max_size=100)
        for i in range(5):
            t = Trajectory()
            t.add_transition(Transition(step=0, action="a", reward=float(i)))
            buf.add(t)
        stats = buf.statistics()
        assert stats["num_episodes"] == 5
        assert stats["mean_reward"] == 2.0
