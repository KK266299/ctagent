# ============================================================================
# 模块职责: IQA 模块单元测试
# ============================================================================
import numpy as np

from src.iqa.metrics import compute_psnr, compute_ssim, compute_metrics


def test_psnr_identical():
    """相同图像 PSNR 应为 inf。"""
    img = np.random.rand(64, 64).astype(np.float32)
    assert compute_psnr(img, img) == float("inf")


def test_psnr_noisy():
    """加噪后 PSNR 应为正有限值。"""
    img = np.random.rand(64, 64).astype(np.float32)
    noisy = img + np.random.normal(0, 0.1, img.shape).astype(np.float32)
    psnr = compute_psnr(noisy, img)
    assert 0 < psnr < 100


def test_ssim_identical():
    """相同图像 SSIM 应为 1.0。"""
    img = np.random.rand(64, 64).astype(np.float32)
    assert abs(compute_ssim(img, img) - 1.0) < 1e-5


def test_compute_metrics():
    """批量指标计算。"""
    img = np.random.rand(64, 64).astype(np.float32)
    metrics = compute_metrics(img, img, ["psnr", "ssim"])
    assert "psnr" in metrics
    assert "ssim" in metrics
