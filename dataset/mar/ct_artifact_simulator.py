from __future__ import annotations

# ============================================================================
# 模块职责: CT 伪影仿真器 — 按与 MARSimulator 一致的架构生成 5 类 CT 伪影
#   输入: 干净 CT (HU) + 几何/物理参数 + severity
#   输出: gt.h5 + <artifact>_<severity>.h5 (含 ma_CT, ma_sinogram, metadata)
#   支持:
#     - ring artifact
#     - motion artifact
#     - beam hardening artifact
#     - scatter artifact
#     - truncation artifact
#     - 多类型组合 / 随机组合
# 参考: Sarepy, TorchIO RandomMotion, LEAP/XrayPhysics, DeepDRR, Hsieh 2004
# ============================================================================

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from dataset.mar.energy_convert import add_poisson_noise, pkev2kvp
from dataset.mar.physics_params import ATTEN_MODE_COL, PhysicsParams
from dataset.mar.sinogram_utils import apply_bhc
from dataset.mar.tissue_decompose import decompose_tissue, hu_to_mu

if TYPE_CHECKING:
    from dataset.mar.ct_geometry import CTGeometry

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ("mild", "moderate", "severe")


def _sample_scalar(value: float | tuple[float, float], rng: np.random.Generator) -> float:
    if isinstance(value, tuple):
        return float(rng.uniform(value[0], value[1]))
    return float(value)


def _sample_int(value: int | tuple[int, int], rng: np.random.Generator) -> int:
    if isinstance(value, tuple):
        lo, hi = value
        return int(rng.integers(lo, hi + 1))
    return int(value)


def _safe_norm(arr: np.ndarray, q: float = 95.0) -> np.ndarray:
    denom = float(np.percentile(arr, q))
    denom = max(denom, 1e-6)
    return np.clip(arr / denom, 0.0, 1.0)


@dataclass
class ArtifactSimulationResult:
    """单张 CT 的伪影仿真结果。"""

    gt_ct: np.ndarray
    poly_sinogram: np.ndarray
    poly_ct: np.ndarray
    artifact_results: list[dict[str, Any]]


@dataclass
class RingArtifactConfig:
    bad_detector_fraction: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.005, 0.010),
        "moderate": (0.010, 0.020),
        "severe": (0.020, 0.030),
    })
    gain_range: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.95, 1.05),
        "moderate": (0.90, 1.10),
        "severe": (0.85, 1.15),
    })
    additive_bias_scale: dict[str, float] = field(default_factory=lambda: {
        "mild": 0.003,
        "moderate": 0.006,
        "severe": 0.010,
    })
    drift_strength: dict[str, float] = field(default_factory=lambda: {
        "mild": 0.002,
        "moderate": 0.004,
        "severe": 0.008,
    })


@dataclass
class MotionArtifactConfig:
    motion_fraction: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.08, 0.12),
        "moderate": (0.12, 0.20),
        "severe": (0.20, 0.30),
    })
    translation_mm: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (1.0, 5.0),
        "moderate": (5.0, 10.0),
        "severe": (10.0, 15.0),
    })
    ghost_blend: dict[str, float] = field(default_factory=lambda: {
        "mild": 0.15,
        "moderate": 0.25,
        "severe": 0.35,
    })


@dataclass
class BeamHardeningArtifactConfig:
    """束硬化伪影配置 — 通过扰动 BHC 多项式系数模拟不完美校正。

    bhc_scale: BHC 系数的缩放因子。<1 表示欠校正 (cupping), >1 表示过校正。
    blend: 混合比例: 0=完全 BHC, 1=完全无 BHC (raw polychromatic)。
    """
    bhc_scale: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.85, 0.95),
        "moderate": (0.65, 0.85),
        "severe": (0.40, 0.65),
    })


@dataclass
class ScatterArtifactConfig:
    scatter_ratio: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.01, 0.03),
        "moderate": (0.03, 0.06),
        "severe": (0.06, 0.10),
    })
    blur_sigma_mm: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (30.0, 45.0),
        "moderate": (45.0, 60.0),
        "severe": (60.0, 80.0),
    })


@dataclass
class TruncationArtifactConfig:
    truncate_ratio: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.05, 0.10),
        "moderate": (0.10, 0.20),
        "severe": (0.20, 0.30),
    })
    min_fraction: dict[str, float] = field(default_factory=lambda: {
        "mild": 0.40,
        "moderate": 0.20,
        "severe": 0.05,
    })
    fill_mode: str = "cosine"


class BaseCTArtifactSimulator:
    """与 MARSimulator 对齐的 2D CT 伪影仿真基类。"""

    artifact_type = "base"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        self.geo = geometry
        self.phy = physics
        self.rng = np.random.default_rng(seed)
        self.minimal = minimal

    def __call__(self, hu_image: np.ndarray, severity: str = "moderate") -> ArtifactSimulationResult:
        return self.simulate(hu_image, severity=severity)

    def simulate(
        self,
        hu_image: np.ndarray,
        severity: str = "moderate",
    ) -> ArtifactSimulationResult:
        if severity not in SEVERITY_LEVELS:
            raise ValueError(f"Unknown severity={severity!r}, valid={SEVERITY_LEVELS}")
        base = self._prepare_base_state(hu_image)
        ma_sinogram, metadata = self._apply_to_sinogram(base, base["poly_sinogram"].copy(), severity)
        artifact_result = self._finalize_result(base, ma_sinogram, severity, metadata)
        return ArtifactSimulationResult(
            gt_ct=base["gt_ct"],
            poly_sinogram=base["poly_sinogram"],
            poly_ct=base["poly_ct"],
            artifact_results=[artifact_result],
        )

    def _prepare_base_state(self, hu_image: np.ndarray) -> dict[str, Any]:
        cfg = self.phy.config

        mu_image = hu_to_mu(hu_image, cfg.mu_water_70kev)
        gt_ct = mu_image.astype(np.float32)

        img_water, img_bone = decompose_tissue(
            mu_image, self.phy.thresh_water, self.phy.thresh_bone
        )

        p_water_kev = self.geo.forward(img_water)
        p_bone_kev = self.geo.forward(img_bone)

        proj_kev_2mat = np.stack([p_water_kev, p_bone_kev], axis=-1)
        mu_list_2 = [self.phy.mu_water, self.phy.mu_bone]
        proj_kvp = pkev2kvp(
            proj_kev_2mat,
            self.phy.spectrum,
            self.phy.energies,
            cfg.kev,
            mu_list_2,
        )
        proj_kvp_noise = add_poisson_noise(
            proj_kvp, cfg.photon_num, cfg.scatter_photon, self.rng
        )
        electronic_sigma = getattr(cfg, "electronic_sigma", 10.0)
        if electronic_sigma > 0:
            proj_kvp_noise = proj_kvp_noise + self.rng.normal(
                0, electronic_sigma / cfg.photon_num, proj_kvp_noise.shape
            )
        poly_sinogram = apply_bhc(proj_kvp_noise, self.phy.para_bhc).astype(np.float32)
        poly_ct = self.geo.fbp(poly_sinogram).astype(np.float32)

        return {
            "hu_image": hu_image.astype(np.float32),
            "mu_image": mu_image.astype(np.float32),
            "gt_ct": gt_ct,
            "img_water": img_water.astype(np.float32),
            "img_bone": img_bone.astype(np.float32),
            "p_water_kev": p_water_kev.astype(np.float32),
            "p_bone_kev": p_bone_kev.astype(np.float32),
            "proj_kvp": proj_kvp.astype(np.float32),
            "proj_kvp_noise": proj_kvp_noise.astype(np.float32),
            "poly_sinogram": poly_sinogram,
            "poly_ct": poly_ct,
        }

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def _finalize_result(
        self,
        base: dict[str, Any],
        ma_sinogram: np.ndarray,
        severity: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        ma_ct = self.geo.fbp(ma_sinogram.astype(np.float32)).astype(np.float32)
        result = {
            "artifact_type": self.artifact_type,
            "severity": severity,
            "ma_CT": ma_ct,
            "ma_sinogram": ma_sinogram.astype(np.float32),
            "params": metadata,
        }
        if not self.minimal:
            result["poly_CT"] = base["poly_ct"].astype(np.float32)
            result["poly_sinogram"] = base["poly_sinogram"].astype(np.float32)
        return result

    @staticmethod
    def save_h5(
        result: ArtifactSimulationResult,
        output_dir: str | Path,
        compression: str = "gzip",
        minimal: bool = False,
    ) -> None:
        import h5py

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        gt_path = out / "gt.h5"
        with h5py.File(str(gt_path), "w") as f:
            f.create_dataset("image", data=result.gt_ct, compression=compression)
            if not minimal:
                f.create_dataset("poly_sinogram", data=result.poly_sinogram, compression=compression)
                f.create_dataset("poly_CT", data=result.poly_ct, compression=compression)

        for item in result.artifact_results:
            fname = f"{item['artifact_type']}_{item['severity']}.h5"
            with h5py.File(str(out / fname), "w") as f:
                f.create_dataset("ma_CT", data=item["ma_CT"], compression=compression)
                f.create_dataset("ma_sinogram", data=item["ma_sinogram"], compression=compression)
                if not minimal:
                    if "poly_CT" in item:
                        f.create_dataset("poly_CT", data=item["poly_CT"], compression=compression)
                    if "poly_sinogram" in item:
                        f.create_dataset("poly_sinogram", data=item["poly_sinogram"], compression=compression)
                f.attrs["artifact_type"] = item["artifact_type"]
                f.attrs["severity"] = item["severity"]
                f.attrs["params_json"] = json.dumps(item.get("params", {}), ensure_ascii=False)


class RingArtifactSimulator(BaseCTArtifactSimulator):
    artifact_type = "ring"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: RingArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or RingArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        views, bins = sinogram.shape
        frac = _sample_scalar(cfg.bad_detector_fraction[severity], self.rng)
        n_bad = max(1, int(round(bins * frac)))
        bad_cols = np.sort(self.rng.choice(bins, size=n_bad, replace=False))

        gain_lo, gain_hi = cfg.gain_range[severity]

        result = sinogram.copy().astype(np.float64)
        sino_std = float(np.std(sinogram))
        bias_scale = cfg.additive_bias_scale[severity] * sino_std
        drift_strength = cfg.drift_strength[severity] * sino_std

        phase = self.rng.uniform(0, 2 * np.pi, size=n_bad)
        drift_curve = np.sin(np.linspace(0, 2 * np.pi, views, dtype=np.float64))[:, None]

        gains = self.rng.uniform(gain_lo, gain_hi, size=n_bad)
        biases = self.rng.uniform(-bias_scale, bias_scale, size=n_bad)
        drift = drift_strength * drift_curve * np.cos(phase)[None, :]

        result[:, bad_cols] = result[:, bad_cols] * gains[None, :] + biases[None, :] + drift
        return result.astype(np.float32), {
            "bad_detector_fraction": round(frac, 5),
            "num_bad_detectors": int(n_bad),
            "gain_range": [float(gain_lo), float(gain_hi)],
            "bad_columns_head": bad_cols[: min(10, len(bad_cols))].tolist(),
            "drift_strength": round(drift_strength, 6),
        }


class MotionArtifactSimulator(BaseCTArtifactSimulator):
    artifact_type = "motion"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: MotionArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or MotionArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        views, bins = sinogram.shape
        motion_fraction = _sample_scalar(cfg.motion_fraction[severity], self.rng)
        span = max(2, int(round(views * motion_fraction)))
        start = int(self.rng.integers(0, max(1, views - span)))
        end = min(views, start + span)

        translation_mm = _sample_scalar(cfg.translation_mm[severity], self.rng)
        shift_bins_max = (translation_mm / 10.0) / max(self.geo.reso, 1e-6)
        ghost_blend = cfg.ghost_blend[severity]

        result = sinogram.copy().astype(np.float64)
        x = np.arange(bins, dtype=np.float64)
        indices = np.arange(start, end, dtype=np.float64)
        ramp = (indices - indices.min()) / max(float(np.ptp(indices)), 1.0)
        signed = np.sign(self.rng.normal()) or 1.0
        shifts = signed * shift_bins_max * ramp

        for row, shift_val in zip(range(start, end), shifts):
            shifted = np.interp(
                x,
                np.clip(x - shift_val, 0.0, float(bins - 1)),
                result[row],
            )
            result[row] = (1.0 - ghost_blend) * shifted + ghost_blend * result[row]

        result[start:end] = gaussian_filter1d(result[start:end], sigma=0.8, axis=0)
        return result.astype(np.float32), {
            "motion_fraction": round(motion_fraction, 4),
            "angle_start_idx": int(start),
            "angle_end_idx": int(end),
            "translation_mm": round(translation_mm, 4),
            "shift_bins_max": round(float(shift_bins_max), 4),
            "ghost_blend": float(ghost_blend),
        }


class BeamHardeningArtifactSimulator(BaseCTArtifactSimulator):
    artifact_type = "beam_hardening"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: BeamHardeningArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or BeamHardeningArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        bhc_scale = _sample_scalar(cfg.bhc_scale[severity], self.rng)

        proj_kvp_noise = base["proj_kvp_noise"].astype(np.float64)
        perturbed_bhc = self.phy.para_bhc * bhc_scale
        result = apply_bhc(proj_kvp_noise, perturbed_bhc)
        return result.astype(np.float32), {
            "bhc_scale": round(bhc_scale, 4),
        }


class ScatterArtifactSimulator(BaseCTArtifactSimulator):
    artifact_type = "scatter"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: ScatterArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or ScatterArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        scatter_ratio = _sample_scalar(cfg.scatter_ratio[severity], self.rng)
        blur_sigma_mm = _sample_scalar(cfg.blur_sigma_mm[severity], self.rng)
        sigma_bins = max((blur_sigma_mm / 10.0) / max(self.geo.reso, 1e-6), 1.0)

        proj_raw = base["proj_kvp_noise"].astype(np.float64)
        primary = np.exp(-proj_raw)
        blurred = gaussian_filter(primary, sigma=(2.0, sigma_bins))
        scatter = scatter_ratio * blurred
        measured = np.clip(primary + scatter, 1e-12, None)
        contaminated = -np.log(measured)
        result = apply_bhc(contaminated, self.phy.para_bhc)
        return result.astype(np.float32), {
            "scatter_ratio": round(scatter_ratio, 6),
            "blur_sigma_mm": round(blur_sigma_mm, 4),
            "blur_sigma_bins": round(float(sigma_bins), 4),
        }


class TruncationArtifactSimulator(BaseCTArtifactSimulator):
    artifact_type = "truncation"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: TruncationArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or TruncationArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        ratio = _sample_scalar(cfg.truncate_ratio[severity], self.rng)
        min_frac = cfg.min_fraction[severity]
        bins = sinogram.shape[1]
        width = max(1, int(round(bins * ratio)))

        result = sinogram.copy().astype(np.float64)
        if cfg.fill_mode == "cosine":
            ramp = min_frac + (1.0 - min_frac) * 0.5 * (
                1.0 - np.cos(np.linspace(0, np.pi, width, dtype=np.float64))
            )
            result[:, :width] *= ramp[None, :]
            result[:, -width:] *= ramp[::-1][None, :]
        else:
            result[:, :width] *= min_frac
            result[:, -width:] *= min_frac

        return result.astype(np.float32), {
            "truncate_ratio": round(ratio, 5),
            "truncate_width_bins": int(width),
            "min_fraction": round(min_frac, 4),
            "fill_mode": cfg.fill_mode,
        }


class CompositeArtifactSimulator(BaseCTArtifactSimulator):
    """组合多个伪影生成器，支持随机选取或顺序叠加。"""

    artifact_type = "composite"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        generators: list[BaseCTArtifactSimulator],
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.generators = generators

    def simulate_composed(
        self,
        hu_image: np.ndarray,
        recipe: list[tuple[BaseCTArtifactSimulator, str]],
    ) -> ArtifactSimulationResult:
        base = self._prepare_base_state(hu_image)
        sinogram = base["poly_sinogram"].copy()
        trace: list[dict[str, Any]] = []
        for gen, severity in recipe:
            sinogram, meta = gen._apply_to_sinogram(base, sinogram, severity)
            trace.append({
                "artifact_type": gen.artifact_type,
                "severity": severity,
                "params": meta,
            })
        ma_ct = self.geo.fbp(sinogram.astype(np.float32)).astype(np.float32)
        return ArtifactSimulationResult(
            gt_ct=base["gt_ct"],
            poly_sinogram=base["poly_sinogram"],
            poly_ct=base["poly_ct"],
            artifact_results=[{
                "artifact_type": "composite",
                "severity": "+".join([s for _, s in recipe]),
                "ma_CT": ma_ct,
                "ma_sinogram": sinogram.astype(np.float32),
                "params": {"recipe": trace},
                "poly_CT": base["poly_ct"],
                "poly_sinogram": base["poly_sinogram"],
            }],
        )

    def simulate_random(
        self,
        hu_image: np.ndarray,
        num_artifacts: int = 2,
        severity: str = "moderate",
    ) -> ArtifactSimulationResult:
        num_artifacts = max(1, min(num_artifacts, len(self.generators)))
        picked = self.rng.choice(self.generators, size=num_artifacts, replace=False)
        recipe = [(gen, severity) for gen in picked.tolist()]
        return self.simulate_composed(hu_image, recipe)

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError("Use simulate_composed() or simulate_random() for composite mode.")


# =========================================================================
# 采样不足类退化
# =========================================================================


@dataclass
class SparseViewArtifactConfig:
    """稀疏视角配置 — num_views 为保留的投影数 (完整通常 640)。"""
    num_views: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "mild": (90, 120),
        "moderate": (45, 90),
        "severe": (20, 45),
    })
    interpolation: str = "linear"


class SparseViewArtifactSimulator(BaseCTArtifactSimulator):
    """稀疏视角伪影: 均匀抽取投影角 → view aliasing / 条纹伪影。

    参考: TAMP create_sparse_view_ct(), Geometry-Aware DRR
    """
    artifact_type = "sparse_view"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: SparseViewArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or SparseViewArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        total_views = sinogram.shape[0]
        num_views = _sample_int(cfg.num_views[severity], self.rng)
        num_views = min(num_views, total_views)

        # 均匀采样视角索引
        indices = np.linspace(0, total_views - 1, num_views, dtype=int)
        result = np.zeros_like(sinogram, dtype=np.float64)
        result[indices] = sinogram[indices]

        # 线性插值填充缺失视角
        if cfg.interpolation == "linear":
            for k in range(len(indices) - 1):
                s, e = int(indices[k]), int(indices[k + 1])
                if e - s <= 1:
                    continue
                for j in range(s + 1, e):
                    alpha = (j - s) / (e - s)
                    result[j] = (1.0 - alpha) * sinogram[s] + alpha * sinogram[e]

        return result.astype(np.float32), {
            "num_views": int(num_views),
            "total_views": int(total_views),
            "subsample_ratio": round(num_views / total_views, 4),
            "interpolation": cfg.interpolation,
        }


@dataclass
class LimitedAngleArtifactConfig:
    """有限角配置 — angle_range_deg 为可用角度范围 (完整 360°)。"""
    angle_range_deg: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (140.0, 160.0),
        "moderate": (100.0, 140.0),
        "severe": (60.0, 100.0),
    })
    transition_fraction: float = 0.1


class LimitedAngleArtifactSimulator(BaseCTArtifactSimulator):
    """有限角伪影: 限制角度范围 → 方向性伪影 + 不适定性。

    参考: EPNet (MICCAI 2021), TAMP create_limited_angle_ct()
    """
    artifact_type = "limited_angle"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: LimitedAngleArtifactConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or LimitedAngleArtifactConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        total_views = sinogram.shape[0]
        angle_range = _sample_scalar(cfg.angle_range_deg[severity], self.rng)
        available = max(1, int(total_views * angle_range / 360.0))
        start = int(self.rng.integers(0, max(1, total_views - available)))
        end = min(start + available, total_views)

        result = np.zeros_like(sinogram, dtype=np.float64)
        result[start:end] = sinogram[start:end]

        # 边界 cosine 过渡 — 减少 Gibbs 效应
        tw = max(1, int(available * cfg.transition_fraction))
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, tw)))
        for i, w in enumerate(ramp):
            if start + i < total_views:
                result[start + i] *= w
        for i, w in enumerate(ramp[::-1]):
            idx = end - tw + i
            if 0 <= idx < total_views:
                result[idx] *= w

        return result.astype(np.float32), {
            "angle_range_deg": round(angle_range, 2),
            "available_views": int(available),
            "start_idx": int(start),
            "end_idx": int(end),
            "transition_width": int(tw),
        }


# =========================================================================
# 低剂量噪声类退化
# =========================================================================


@dataclass
class LowDoseNoiseConfig:
    """低剂量噪声配置 — dose_fraction 为相对满剂量的比例。"""
    dose_fraction: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.20, 0.50),
        "moderate": (0.05, 0.20),
        "severe": (0.01, 0.05),
    })
    electronic_sigma: dict[str, float] = field(default_factory=lambda: {
        "mild": 5.0,
        "moderate": 10.0,
        "severe": 20.0,
    })


class LowDoseNoiseSimulator(BaseCTArtifactSimulator):
    """低剂量噪声: 在 sinogram 域按剂量比例重新采样 Poisson + Gaussian 噪声。

    注意: base pipeline 已在满剂量下加过 Poisson 噪声，此处将 sinogram
    转回透射域后以更低光子数重新采样，模拟低剂量采集。

    参考: LD-CT-simulation, RTK AddNoise, XCIST quantum+electronic noise
    """
    artifact_type = "low_dose"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: LowDoseNoiseConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or LowDoseNoiseConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        dose_frac = _sample_scalar(cfg.dose_fraction[severity], self.rng)
        e_sigma = cfg.electronic_sigma[severity]
        photon_num = self.phy.config.photon_num * dose_frac

        proj_kvp = base["proj_kvp"].astype(np.float64)

        noisy = add_poisson_noise(
            proj_kvp, photon_num, self.phy.config.scatter_photon, self.rng
        )
        noisy += self.rng.normal(0, e_sigma / max(photon_num, 1.0), noisy.shape)
        result = apply_bhc(noisy, self.phy.para_bhc)
        return result.astype(np.float32), {
            "dose_fraction": round(dose_frac, 4),
            "effective_photon_num": float(photon_num),
            "electronic_sigma": float(e_sigma),
        }


# =========================================================================
# 探测器 / 系统分辨率类退化
# =========================================================================


@dataclass
class FocalSpotBlurConfig:
    """焦点模糊配置 — sigma 为探测器方向 (bins) 上的高斯模糊宽度。"""
    blur_sigma_bins: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.5, 1.0),
        "moderate": (1.0, 2.0),
        "severe": (2.0, 4.0),
    })
    axial_sigma_views: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.0, 0.3),
        "moderate": (0.3, 0.8),
        "severe": (0.8, 1.5),
    })


class FocalSpotBlurSimulator(BaseCTArtifactSimulator):
    """焦点 / 探测器模糊: 沿探测器方向 + 可选轴向模糊。

    模拟 focal spot size、detector cell width、detector crosstalk
    导致的空间分辨率退化。

    参考: XCIST focal spot / detector column width / optical crosstalk
    """
    artifact_type = "focal_spot_blur"

    def __init__(
        self,
        geometry: CTGeometry,
        physics: PhysicsParams,
        config: FocalSpotBlurConfig | None = None,
        seed: int | None = None,
        minimal: bool = False,
    ) -> None:
        super().__init__(geometry, physics, seed=seed, minimal=minimal)
        self.config = config or FocalSpotBlurConfig()

    def _apply_to_sinogram(
        self,
        base: dict[str, Any],
        sinogram: np.ndarray,
        severity: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        sigma_bins = _sample_scalar(cfg.blur_sigma_bins[severity], self.rng)
        sigma_views = _sample_scalar(cfg.axial_sigma_views[severity], self.rng)

        result = sinogram.astype(np.float64)
        # 探测器方向 (axis=1) 模糊 — focal spot + detector cell
        if sigma_bins > 0.01:
            result = gaussian_filter1d(result, sigma=sigma_bins, axis=1)
        # 视角方向 (axis=0) 轻微模糊 — 旋转采样间隔效应
        if sigma_views > 0.01:
            result = gaussian_filter1d(result, sigma=sigma_views, axis=0)

        return result.astype(np.float32), {
            "blur_sigma_bins": round(sigma_bins, 4),
            "axial_sigma_views": round(sigma_views, 4),
        }


# =========================================================================
# 注册表 & 工厂
# =========================================================================


ARTIFACT_SIMULATOR_REGISTRY = {
    "ring": RingArtifactSimulator,
    "motion": MotionArtifactSimulator,
    "beam_hardening": BeamHardeningArtifactSimulator,
    "scatter": ScatterArtifactSimulator,
    "truncation": TruncationArtifactSimulator,
    "sparse_view": SparseViewArtifactSimulator,
    "limited_angle": LimitedAngleArtifactSimulator,
    "low_dose": LowDoseNoiseSimulator,
    "focal_spot_blur": FocalSpotBlurSimulator,
}


def create_artifact_simulator(
    artifact_type: str,
    geometry: CTGeometry,
    physics: PhysicsParams,
    seed: int | None = None,
    minimal: bool = False,
) -> BaseCTArtifactSimulator:
    artifact_type = artifact_type.strip().lower()
    cls = ARTIFACT_SIMULATOR_REGISTRY.get(artifact_type)
    if cls is None:
        raise ValueError(
            f"Unknown artifact_type={artifact_type!r}, valid={sorted(ARTIFACT_SIMULATOR_REGISTRY)}"
        )
    return cls(geometry, physics, seed=seed, minimal=minimal)
