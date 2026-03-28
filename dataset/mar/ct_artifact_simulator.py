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
    alpha: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.05, 0.08),
        "moderate": (0.08, 0.12),
        "severe": (0.12, 0.15),
    })
    beta: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "mild": (0.001, 0.003),
        "moderate": (0.003, 0.006),
        "severe": (0.006, 0.010),
    })
    bone_emphasis_power: float = 1.5


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
        alpha = _sample_scalar(cfg.alpha[severity], self.rng)
        beta = _sample_scalar(cfg.beta[severity], self.rng)

        bone_weight = _safe_norm(base["p_bone_kev"]) ** cfg.bone_emphasis_power
        result = sinogram.astype(np.float64).copy()
        result = result + bone_weight * (alpha * result**2 + beta * result**3)
        return result.astype(np.float32), {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "bone_emphasis_power": float(cfg.bone_emphasis_power),
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

        primary = np.exp(-sinogram.astype(np.float64))
        blurred = gaussian_filter(primary, sigma=(2.0, sigma_bins))
        scatter = scatter_ratio * blurred
        measured = np.clip(primary + scatter, 1e-12, None)
        result = -np.log(measured)
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
        bins = sinogram.shape[1]
        width = max(1, int(round(bins * ratio)))

        result = sinogram.copy().astype(np.float64)
        if cfg.fill_mode == "cosine":
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, width, dtype=np.float64)))
            result[:, :width] *= ramp[None, :]
            result[:, -width:] *= ramp[::-1][None, :]
        else:
            result[:, :width] = 0.0
            result[:, -width:] = 0.0

        return result.astype(np.float32), {
            "truncate_ratio": round(ratio, 5),
            "truncate_width_bins": int(width),
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


ARTIFACT_SIMULATOR_REGISTRY = {
    "ring": RingArtifactSimulator,
    "motion": MotionArtifactSimulator,
    "beam_hardening": BeamHardeningArtifactSimulator,
    "scatter": ScatterArtifactSimulator,
    "truncation": TruncationArtifactSimulator,
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
