# ============================================================================
# 模块职责: Toy CT Phantom 生成器 — 无真实数据时的最小验证数据源
#   生成 2D CT-like phantom:
#     - 均匀背景 (模拟空气/软组织)
#     - 1~3 个大椭圆结构 (模拟器官)
#     - 0~2 个 subtle 高亮区域 (模拟 lesion)
#   lesion 对比度刻意设为 subtle (intensity 0.55-0.65 vs organ 0.35-0.55)
#   以确保噪声能真正干扰诊断，restoration 有实际价值
# 参考: MedQ-Bench — synthetic phantom for evaluation
#       CAPIQA — CT quality assessment phantom
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from src.degradations.simulator import DegradationSimulator
from src.degradations.types import DegradationType


@dataclass
class ToyLabel:
    """Toy phantom 的 ground truth label。"""
    lesion_present: bool = False
    lesion_count: int = 0
    lesion_positions: list[dict[str, Any]] = field(default_factory=list)
    organ_count: int = 0
    image_size: tuple[int, int] = (256, 256)
    degradation_type: str = "none"
    degradation_params: dict[str, Any] = field(default_factory=dict)

    @property
    def lesion_side(self) -> str:
        if not self.lesion_positions:
            return "none"
        cx = self.image_size[1] // 2
        sides = ["left" if p["x"] < cx else "right" for p in self.lesion_positions]
        if all(s == "left" for s in sides):
            return "left"
        if all(s == "right" for s in sides):
            return "right"
        return "bilateral"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lesion_side"] = self.lesion_side
        return d


def generate_toy_phantom(
    size: int = 256,
    num_organs: int = 2,
    num_lesions: int | None = None,
    lesion_intensity: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, ToyLabel]:
    """生成一个 2D CT-like phantom。

    Lesion 刻意设为 subtle:
    - intensity 仅比周围器官高 0.05-0.15
    - radius 3-7 像素 (在 256x256 上非常小)
    - 噪声 sigma=0.08 足以淹没这种差异

    Returns:
        (image, label): image 值域 [0, 1], label 含 ground truth
    """
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 0.1, dtype=np.float32)

    if num_lesions is None:
        num_lesions = rng.choice([0, 0, 1, 1, 2], p=[0.2, 0.2, 0.25, 0.25, 0.1])

    label = ToyLabel(image_size=(size, size), organ_count=num_organs)

    yy, xx = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # 大椭圆 — 身体轮廓
    body_a, body_b = size * 0.4, size * 0.35
    body_mask = ((yy - cy) / body_a) ** 2 + ((xx - cx) / body_b) ** 2 <= 1.0
    image[body_mask] = 0.3

    # 器官 — 中等椭圆, intensity 0.35-0.50
    organ_regions: list[tuple[int, int, float]] = []
    for _ in range(num_organs):
        ox = cx + rng.integers(-size // 5, size // 5)
        oy = cy + rng.integers(-size // 6, size // 6)
        oa = rng.integers(size // 8, size // 5)
        ob = rng.integers(size // 8, size // 5)
        organ_mask = ((yy - oy) / max(oa, 1)) ** 2 + ((xx - ox) / max(ob, 1)) ** 2 <= 1.0
        organ_val = rng.uniform(0.35, 0.50)
        image[organ_mask] = organ_val
        organ_regions.append((int(oy), int(ox), organ_val))

    # Lesion — subtle 高亮, 仅比周围高 0.08-0.15
    for _ in range(num_lesions):
        lx = cx + rng.integers(-size // 4, size // 4)
        ly = cy + rng.integers(-size // 4, size // 4)
        lr = rng.integers(max(3, size // 80), max(5, size // 40))

        local_bg = float(image[
            max(0, ly - lr * 2):min(size, ly + lr * 2),
            max(0, lx - lr * 2):min(size, lx + lr * 2),
        ].mean())

        if lesion_intensity is not None:
            l_val = lesion_intensity
        else:
            l_val = local_bg + rng.uniform(0.08, 0.15)
            l_val = min(l_val, 0.70)

        lesion_mask = (yy - ly) ** 2 + (xx - lx) ** 2 <= lr ** 2
        image[lesion_mask] = l_val

        label.lesion_positions.append({
            "y": int(ly), "x": int(lx), "radius": int(lr),
            "intensity": round(float(l_val), 3),
            "local_background": round(local_bg, 3),
            "contrast": round(float(l_val - local_bg), 3),
        })

    label.lesion_present = num_lesions > 0
    label.lesion_count = num_lesions

    return image, label


def generate_toy_case(
    size: int = 256,
    degradation: str = "noise",
    degradation_params: dict[str, Any] | None = None,
    seed: int | None = None,
    **phantom_kwargs: Any,
) -> dict[str, Any]:
    """生成一个完整的 toy case: clean + degraded + label。"""
    clean, label = generate_toy_phantom(size=size, seed=seed, **phantom_kwargs)

    deg_map = {
        "noise": DegradationType.NOISE,
        "blur": DegradationType.BLUR,
        "low_resolution": DegradationType.LOW_RESOLUTION,
    }
    deg_type = deg_map.get(degradation, DegradationType.NOISE)
    params = degradation_params or {}
    if degradation == "noise" and "sigma" not in params:
        params["sigma"] = 0.08

    simulator = DegradationSimulator()
    degraded = simulator.apply(clean, deg_type, **params)
    degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)

    label.degradation_type = degradation
    label.degradation_params = params

    case_id = f"toy_{seed if seed is not None else 'rand'}_{degradation}"
    return {
        "clean": clean,
        "degraded": degraded,
        "label": label,
        "case_id": case_id,
    }
