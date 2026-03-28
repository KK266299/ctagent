#!/usr/bin/env python
"""CT 伪影生成器可视化验证 — 在真实 CQ500 CT 影像上施加 5 类伪影。

参考: visualize_mar_results.py / visualize_restoration.py 的风格
输出: 多 panel 对比图 (brain + subdural 窗), 差异图, 正弦图对比

用法:
    PYTHONPATH=. bash -c '
    eval "$(conda shell.bash hook)" && conda activate llamafactory && \
    CUDA_VISIBLE_DEVICES=1 python -u scripts/test_ct_artifact_generators.py \
        --output-dir /home/liuxinyao/project/ctagent/try_output \
        --gpu 1 --num-slices 3'
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset.mar.ct_artifact_simulator import (
    ARTIFACT_SIMULATOR_REGISTRY,
    BeamHardeningArtifactSimulator,
    CompositeArtifactSimulator,
    MotionArtifactSimulator,
    RingArtifactSimulator,
    ScatterArtifactSimulator,
    TruncationArtifactSimulator,
    create_artifact_simulator,
)
from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig
from dataset.mar.physics_params import ATTEN_MODE_COL, PhysicsConfig, PhysicsParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vis_ct_artifacts")


MU_WATER = 0.192


def mu_to_hu(mu: np.ndarray) -> np.ndarray:
    return (mu / MU_WATER - 1.0) * 1000.0


def hu_window_to_mu(wl_hu: float, ww_hu: float) -> tuple[float, float]:
    hu_lo = wl_hu - ww_hu / 2.0
    hu_hi = wl_hu + ww_hu / 2.0
    return MU_WATER * (1.0 + hu_lo / 1000.0), MU_WATER * (1.0 + hu_hi / 1000.0)


WINDOWS = {
    "brain": (40, 80),
    "subdural": (75, 215),
}


def mu_to_display(img: np.ndarray, window: str = "brain") -> np.ndarray:
    wl, ww = WINDOWS.get(window, (40, 80))
    lo, hi = hu_window_to_mu(wl, ww)
    return np.clip((img - lo) / max(hi - lo, 1e-10), 0.0, 1.0)


def find_real_slices(
    processed_dir: str,
    num_slices: int = 3,
) -> list[Path]:
    """从 cq500_processed 中找中间层 gt.h5（脑组织最丰富）。"""
    root = Path(processed_dir)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    selected: list[Path] = []
    for case_dir in cases:
        if len(selected) >= num_slices:
            break
        series_dirs = sorted([d for d in case_dir.iterdir() if d.is_dir()])
        for series_dir in series_dirs:
            slice_dirs = sorted([d for d in series_dir.iterdir() if d.is_dir()])
            if len(slice_dirs) < 5:
                continue
            mid = len(slice_dirs) // 2
            gt_path = slice_dirs[mid] / "gt.h5"
            if gt_path.exists():
                selected.append(gt_path)
                break
    if not selected:
        raise FileNotFoundError(f"No suitable gt.h5 found in {root}")
    return selected


def load_mu_from_h5(gt_path: Path) -> np.ndarray:
    with h5py.File(str(gt_path), "r") as f:
        return f["image"][:].astype(np.float64)


def make_physics(mat_dir: str = "data/mar_physics") -> PhysicsParams:
    """加载真实 .mat 物理参数。"""
    cfg = PhysicsConfig(mat_dir=mat_dir)
    phy = PhysicsParams(cfg)
    phy.load()
    return phy


def plot_artifact_gallery(
    gt_mu: np.ndarray,
    artifacts: dict[str, np.ndarray],
    case_label: str,
    window: str,
    output_path: Path,
) -> None:
    """生成 1 + N panel 对比图: GT | artifact_1 | artifact_2 | ..."""
    n = 1 + len(artifacts)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    gt_disp = mu_to_display(gt_mu, window)
    axes[0].imshow(gt_disp, cmap="gray")
    axes[0].set_title("GT (clean)", fontsize=11)
    axes[0].axis("off")

    for idx, (label, art_mu) in enumerate(artifacts.items(), 1):
        art_disp = mu_to_display(art_mu, window)
        axes[idx].imshow(art_disp, cmap="gray")
        axes[idx].set_title(label, fontsize=10)
        axes[idx].axis("off")

    fig.suptitle(f"{case_label}  [{window} window]", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_severity_row(
    gt_mu: np.ndarray,
    mild_mu: np.ndarray,
    moderate_mu: np.ndarray,
    severe_mu: np.ndarray,
    artifact_type: str,
    case_label: str,
    window: str,
    output_path: Path,
) -> None:
    """GT | mild | moderate | severe 四 panel 对比。"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        ("GT (clean)", gt_mu),
        (f"{artifact_type}/mild", mild_mu),
        (f"{artifact_type}/moderate", moderate_mu),
        (f"{artifact_type}/severe", severe_mu),
    ]
    for ax, (title, mu) in zip(axes, panels):
        ax.imshow(mu_to_display(mu, window), cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"{case_label}  [{window} window]", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_diff(
    gt_mu: np.ndarray,
    artifact_mus: dict[str, np.ndarray],
    case_label: str,
    output_path: Path,
) -> None:
    """差异图: |artifact - GT|，用 hot colormap。"""
    n = len(artifact_mus)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (label, art_mu) in zip(axes, artifact_mus.items()):
        diff = np.abs(art_mu - gt_mu)
        im = ax.imshow(diff, cmap="hot", vmin=0, vmax=0.10)
        ax.set_title(f"|{label} − GT|", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Error Maps — {case_label}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CT artifact generators on real CQ500")
    parser.add_argument("--output-dir", default="try_output")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--processed-dir", default="/home/liuxinyao/data/cq500_processed")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing CT geometry (416×416, 640 views, astra_cuda) ...")
    geo = CTGeometry(
        CTGeometryConfig(
            image_size=416,
            num_angles=640,
            num_detectors=641,
            impl="astra_cuda",
        )
    )
    phy = make_physics()

    gt_paths = find_real_slices(args.processed_dir, args.num_slices)
    logger.info("Selected %d real CT slices", len(gt_paths))

    artifact_types = ["ring", "motion", "beam_hardening", "scatter", "truncation"]

    for si, gt_path in enumerate(gt_paths):
        rel = gt_path.parent.relative_to(args.processed_dir)
        case_label = str(rel).replace("/", "_")
        logger.info("[%d/%d] Loading %s ...", si + 1, len(gt_paths), case_label)

        mu_image = load_mu_from_h5(gt_path)
        hu_image = mu_to_hu(mu_image)

        # --- 1. Gallery: poly_ct (clean recon) + 5 类 moderate 伪影 ---
        moderate_results: dict[str, np.ndarray] = {}
        clean_recon = None
        for art_type in artifact_types:
            sim = create_artifact_simulator(art_type, geo, phy, seed=42 + si)
            result = sim.simulate(hu_image, severity="moderate")
            moderate_results[art_type] = result.artifact_results[0]["ma_CT"]
            if clean_recon is None:
                clean_recon = result.poly_ct

        for win in ("brain", "subdural"):
            gallery_path = out_dir / f"{case_label}_gallery_{win}.png"
            plot_artifact_gallery(clean_recon, moderate_results, case_label, win, gallery_path)
            logger.info("  Saved gallery %s", gallery_path.name)

        # --- 2. 差异图 ---
        diff_path = out_dir / f"{case_label}_diff.png"
        plot_diff(clean_recon, moderate_results, case_label, diff_path)
        logger.info("  Saved diff %s", diff_path.name)

        # --- 3. 每类伪影 severity 递进 ---
        for art_type in artifact_types:
            severity_mus: dict[str, np.ndarray] = {}
            for sev in ("mild", "moderate", "severe"):
                sim = create_artifact_simulator(art_type, geo, phy, seed=42 + si)
                result = sim.simulate(hu_image, severity=sev)
                severity_mus[sev] = result.artifact_results[0]["ma_CT"]

            for win in ("brain", "subdural"):
                sev_path = out_dir / f"{case_label}_{art_type}_severity_{win}.png"
                plot_severity_row(
                    clean_recon,
                    severity_mus["mild"],
                    severity_mus["moderate"],
                    severity_mus["severe"],
                    art_type,
                    case_label,
                    win,
                    sev_path,
                )
            logger.info("  Saved %s severity row", art_type)

    logger.info("All done! Images saved to %s", out_dir)


if __name__ == "__main__":
    main()
