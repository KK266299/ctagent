# ============================================================================
# 模块职责: I/O 模块单元测试
# ============================================================================
import numpy as np

from src.io.windowing import apply_window, WINDOW_PRESETS


def test_window_range():
    """窗口化结果应在 [0, 1]。"""
    img = np.linspace(-1000, 1000, 100).astype(np.float32)
    result = apply_window(img, center=40, width=400, normalize=True)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_window_presets():
    """预设窗口应存在常用类型。"""
    assert "soft_tissue" in WINDOW_PRESETS
    assert "lung" in WINDOW_PRESETS
    assert "bone" in WINDOW_PRESETS
