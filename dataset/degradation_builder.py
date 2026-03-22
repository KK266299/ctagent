# ============================================================================
# 模块职责: Degradation Builder — 从 clean slices 批量生成 degraded slices
#   参考 ADN 的 artifact synthesis 思路, 但扩展为多类型可控退化:
#     - noise: Gaussian/Poisson 噪声 (模拟 low-dose CT)
#     - blur: Gaussian 模糊 (模拟运动/散焦)
#     - downsample: 下采样+上采样 (模拟分辨率退化)
#     - artifact: 条纹/环形伪影 (简化版)
#   每种退化支持 severity 级别 (1~5)
#   从 clean npy 读取 → 施加退化 → 存为 degraded npy → 记入 manifest
# 参考: ADN — artifact synthesis + prepare_spineweb.py 的数据构建模式
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dataset.manifest import read_manifest, write_manifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 退化参数预设 — severity 1(轻) ~ 5(重)
# ---------------------------------------------------------------------------

NOISE_PARAMS: dict[int, dict[str, float]] = {
    1: {"sigma": 0.02},
    2: {"sigma": 0.05},
    3: {"sigma": 0.08},
    4: {"sigma": 0.12},
    5: {"sigma": 0.18},
}

BLUR_PARAMS: dict[int, dict[str, float]] = {
    1: {"sigma": 0.5},
    2: {"sigma": 1.0},
    3: {"sigma": 1.5},
    4: {"sigma": 2.0},
    5: {"sigma": 3.0},
}

DOWNSAMPLE_PARAMS: dict[int, dict[str, Any]] = {
    1: {"factor": 1.5},
    2: {"factor": 2.0},
    3: {"factor": 3.0},
    4: {"factor": 4.0},
    5: {"factor": 6.0},
}

ARTIFACT_PARAMS: dict[int, dict[str, Any]] = {
    1: {"num_streaks": 2, "intensity": 0.03},
    2: {"num_streaks": 4, "intensity": 0.05},
    3: {"num_streaks": 6, "intensity": 0.08},
    4: {"num_streaks": 8, "intensity": 0.12},
    5: {"num_streaks": 12, "intensity": 0.18},
}

DEFAULT_PARAMS: dict[str, dict[int, dict[str, Any]]] = {
    "noise": NOISE_PARAMS,
    "blur": BLUR_PARAMS,
    "downsample": DOWNSAMPLE_PARAMS,
    "artifact": ARTIFACT_PARAMS,
}


# ---------------------------------------------------------------------------
# 退化函数
# ---------------------------------------------------------------------------

def degrade_noise(image: np.ndarray, sigma: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noisy = image + rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def degrade_blur(image: np.ndarray, sigma: float = 1.0, **_: Any) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(image, sigma=sigma)
    return blurred.astype(np.float32)


def degrade_downsample(image: np.ndarray, factor: float = 2.0, **_: Any) -> np.ndarray:
    from PIL import Image as PILImage
    h, w = image.shape[:2]
    small_h, small_w = max(1, int(h / factor)), max(1, int(w / factor))
    pil = PILImage.fromarray((image * 255).astype(np.uint8))
    small = pil.resize((small_w, small_h), PILImage.BILINEAR)
    back = small.resize((w, h), PILImage.BILINEAR)
    return np.array(back).astype(np.float32) / 255.0


def degrade_artifact(
    image: np.ndarray,
    num_streaks: int = 4,
    intensity: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """简化版条纹伪影: 随机方向线条叠加。"""
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    artifact = np.zeros_like(image)
    for _ in range(num_streaks):
        if rng.random() > 0.5:
            row = rng.integers(0, h)
            thickness = rng.integers(1, 4)
            r_start = max(0, row - thickness)
            r_end = min(h, row + thickness)
            artifact[r_start:r_end, :] = intensity * (0.5 + rng.random())
        else:
            col = rng.integers(0, w)
            thickness = rng.integers(1, 4)
            c_start = max(0, col - thickness)
            c_end = min(w, col + thickness)
            artifact[:, c_start:c_end] = intensity * (0.5 + rng.random())
    degraded = image + artifact
    return np.clip(degraded, 0.0, 1.0).astype(np.float32)


DEGRADE_FN = {
    "noise": degrade_noise,
    "blur": degrade_blur,
    "downsample": degrade_downsample,
    "artifact": degrade_artifact,
}


# ---------------------------------------------------------------------------
# 批量构建
# ---------------------------------------------------------------------------

@dataclass
class DegradationConfig:
    """单种退化的配置。"""
    degradation_type: str
    severities: list[int] = field(default_factory=lambda: [1, 2, 3])
    params_override: dict[int, dict[str, Any]] | None = None


def build_degraded_dataset(
    clean_manifest_path: str | Path,
    output_dir: str | Path,
    degraded_manifest_path: str | Path,
    configs: list[DegradationConfig],
    seed: int = 42,
    max_slices: int | None = None,
) -> int:
    """从 clean slices 批量生成 degraded slices。

    Returns:
        生成的 degraded slice 总数
    """
    clean_records = read_manifest(clean_manifest_path)
    if max_slices:
        clean_records = clean_records[:max_slices]

    logger.info("Building degraded dataset from %d clean slices", len(clean_records))

    out_root = Path(output_dir)
    rng = np.random.default_rng(seed)
    degraded_records: list[dict[str, Any]] = []
    total = 0

    for cfg in configs:
        fn = DEGRADE_FN.get(cfg.degradation_type)
        if fn is None:
            logger.warning("Unknown degradation type: %s, skipping", cfg.degradation_type)
            continue

        param_presets = cfg.params_override or DEFAULT_PARAMS.get(cfg.degradation_type, {})

        for severity in cfg.severities:
            params = param_presets.get(severity, {})
            sev_dir = out_root / cfg.degradation_type / f"severity_{severity}"
            sev_dir.mkdir(parents=True, exist_ok=True)

            for rec in clean_records:
                clean_path = rec["npy_path"]
                try:
                    clean_img = np.load(clean_path)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", clean_path, e)
                    continue

                kwargs = dict(params)
                if cfg.degradation_type in ("noise", "artifact"):
                    kwargs["rng"] = rng

                degraded_img = fn(clean_img, **kwargs)

                slice_id = rec.get("slice_id", Path(clean_path).stem)
                npy_name = f"{slice_id}.npy"
                npy_path = sev_dir / npy_name
                np.save(str(npy_path), degraded_img)

                degraded_records.append({
                    "slice_id": slice_id,
                    "patient_id": rec.get("patient_id", ""),
                    "series_uid": rec.get("series_uid", ""),
                    "degradation_type": cfg.degradation_type,
                    "severity": severity,
                    "params": params,
                    "clean_path": clean_path,
                    "degraded_path": str(npy_path),
                })
                total += 1

            logger.info("  %s severity=%d: %d slices", cfg.degradation_type, severity, len(clean_records))

    write_manifest(degraded_records, degraded_manifest_path)
    logger.info("Degraded dataset: %d total slices → %s", total, out_root)
    return total
