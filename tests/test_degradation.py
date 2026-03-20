# ============================================================================
# 模块职责: 退化模块单元测试
# ============================================================================
import numpy as np

from src.degradations.detector import DegradationDetector
from src.degradations.simulator import DegradationSimulator
from src.degradations.types import DegradationType


def test_detector_clean_image():
    """干净图像不应检出严重退化。"""
    detector = DegradationDetector()
    # 平滑图像, 噪声极低
    img = np.ones((64, 64), dtype=np.float32) * 0.5
    report = detector.detect(img)
    assert len(report.degradations) == 0


def test_detector_noisy_image():
    """高噪声图像应检出噪声退化。"""
    detector = DegradationDetector()
    img = np.random.normal(0, 60, (64, 64)).astype(np.float32)
    report = detector.detect(img)
    assert len(report.degradations) > 0


def test_simulator_noise():
    """噪声模拟应改变图像。"""
    simulator = DegradationSimulator()
    img = np.zeros((64, 64), dtype=np.float32)
    noisy = simulator.apply(img, DegradationType.NOISE, sigma=25)
    assert not np.allclose(img, noisy)


def test_simulator_blur():
    """模糊模拟应改变图像。"""
    simulator = DegradationSimulator()
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    blurred = simulator.apply(img, DegradationType.BLUR)
    assert not np.array_equal(img, blurred)
