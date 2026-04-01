#!/usr/bin/env python
"""可视化全部 10 类退化类型 (含金属伪影)。

从 CQ500 取中间切片，生成 10 类退化 × 3 级 severity 的对比图。
每类退化一行：GT + mild + moderate + severe。

用法:
    PYTHONPATH=. bash -c '
    eval "$(conda shell.bash hook)" && conda activate llamafactory && \
    CUDA_VISIBLE_DEVICES=1 python -u scripts/visualize_all_degradations.py \
        --output-dir try_output/degradation_review \
        --gpu 1'
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import scipy.io as sio

MU_WATER = 0.192
WINDOWS = {"brain": (40, 80), "subdural": (75, 215)}

ARTIFACT_LABELS = {
    "metal":          "1. Metal Artifact\n   (金属伪影)",
    "ring":           "2. Ring Artifact\n   (环状伪影)",
    "motion":         "3. Motion Artifact\n   (运动伪影)",
    "beam_hardening": "4. Beam Hardening\n   (束硬化伪影)",
    "scatter":        "5. Scatter Artifact\n   (散射伪影)",
    "truncation":     "6. Truncation Artifact\n   (截断伪影)",
    "low_dose":       "7. Low-Dose Noise\n   (低剂量噪声)",
    "sparse_view":    "8. Sparse-View\n   (稀疏角采样)",
    "limited_angle":  "9. Limited-Angle\n   (有限角采样)",
    "focal_spot_blur":"10. Focal Spot Blur\n    (焦点模糊)",
}

ARTIFACT_PHYSICS = {
    "metal":          "Photon starvation + BHC residual",
    "ring":           "Detector gain/bias anomaly",
    "motion":         "Patient translation during scan",
    "beam_hardening": "Incomplete BHC polynomial correction",
    "scatter":        "Low-freq scatter in transmission domain",
    "truncation":     "FOV edge attenuation loss",
    "low_dose":       "Reduced I₀ → increased Poisson variance",
    "sparse_view":    "Uniform angular undersampling",
    "limited_angle":  "Missing angular range (Radon wedge)",
    "focal_spot_blur":"Detector/focal-spot PSF blurring",
}


def hu_window_to_mu(wl_hu: float, ww_hu: float) -> tuple[float, float]:
    hu_lo = wl_hu - ww_hu / 2.0
    hu_hi = wl_hu + ww_hu / 2.0
    return MU_WATER * (1.0 + hu_lo / 1000.0), MU_WATER * (1.0 + hu_hi / 1000.0)


def mu_to_display(img: np.ndarray, window: str = "brain") -> np.ndarray:
    wl, ww = WINDOWS.get(window, (40, 80))
    lo, hi = hu_window_to_mu(wl, ww)
    return np.clip((img - lo) / max(hi - lo, 1e-10), 0.0, 1.0)


def mu_to_hu(mu: np.ndarray) -> np.ndarray:
    return (mu / MU_WATER - 1.0) * 1000.0


def find_middle_slice(processed_dir: str) -> Path | None:
    root = Path(processed_dir)
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("."):
            continue
        for series_dir in sorted(case_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            slice_dirs = sorted([d for d in series_dir.iterdir() if d.is_dir()])
            if len(slice_dirs) < 5:
                continue
            mid = len(slice_dirs) // 2
            gt_path = slice_dirs[mid] / "gt.h5"
            if gt_path.exists():
                return gt_path
    return None


def load_metal_masks(mask_mat_path: str, num: int = 3) -> list[np.ndarray]:
    mat = sio.loadmat(mask_mat_path)
    masks_all = mat["CT_samples_bwMetal"]
    picked = []
    total = masks_all.shape[2]
    for i in [0, total // 3, total * 2 // 3]:
        picked.append(masks_all[:, :, min(i, total - 1)].astype(np.float32))
    return picked[:num]


def simulate_metal(hu_image, geo, phy, metal_masks):
    from dataset.mar.mar_simulator import MARSimulator
    sim = MARSimulator(geo, phy, seed=42)
    result = sim.simulate(hu_image, metal_masks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="try_output/degradation_review")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--processed-dir", default="/home/liuxinyao/data/cq500_processed")
    parser.add_argument("--mat-dir", default="data/mar_physics")
    parser.add_argument("--mask-mat", default="data/mar_physics/SampleMasks.mat")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from dataset.mar.ct_artifact_simulator import create_artifact_simulator
    from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig
    from dataset.mar.physics_params import PhysicsConfig, PhysicsParams
    from src.iqa.metrics import compute_psnr, compute_ssim

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing CT geometry ...")
    geo = CTGeometry(CTGeometryConfig(
        image_size=416, num_angles=640, num_detectors=641, impl="astra_cuda",
    ))
    phy_cfg = PhysicsConfig(mat_dir=args.mat_dir)
    phy = PhysicsParams(phy_cfg)
    phy.load()

    gt_path = find_middle_slice(args.processed_dir)
    if gt_path is None:
        print("ERROR: No valid slice found")
        sys.exit(1)

    rel = gt_path.parent.relative_to(args.processed_dir)
    case_label = str(rel).replace("/", "_")
    print(f"Using slice: {case_label}")

    with h5py.File(str(gt_path), "r") as f:
        mu_image = f["image"][:].astype(np.float64)
    hu_image = mu_to_hu(mu_image)

    metal_masks = load_metal_masks(args.mask_mat, num=3)
    print(f"Loaded {len(metal_masks)} metal masks")

    artifact_types = list(ARTIFACT_LABELS.keys())
    severities = ["mild", "moderate", "severe"]
    n_rows = len(artifact_types)

    mar_result = simulate_metal(hu_image, geo, phy, metal_masks)
    print("MAR simulation done")

    for window in ("brain", "subdural"):
        gt_disp = mu_to_display(mu_image, window)
        poly_ct_disp = mu_to_display(mar_result.poly_ct, window)

        fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.2 * n_rows))
        fig.subplots_adjust(hspace=0.35, wspace=0.08)

        for row, art_type in enumerate(artifact_types):
            # Column 0: GT
            axes[row, 0].imshow(poly_ct_disp if art_type != "metal" else gt_disp, cmap="gray")
            axes[row, 0].set_title("GT (poly_ct)" if art_type != "metal" else "GT (μ image)", fontsize=10)
            axes[row, 0].set_ylabel(
                ARTIFACT_LABELS[art_type],
                fontsize=10, fontweight="bold", rotation=0, labelpad=140, va="center",
            )
            axes[row, 0].axis("off")

            if art_type == "metal":
                for col in range(3):
                    ax = axes[row, col + 1]
                    if col < len(mar_result.mask_results):
                        mr = mar_result.mask_results[col]
                        ma_ct = mr["ma_CT"]
                        data_range = float(max(mar_result.poly_ct.max() - mar_result.poly_ct.min(), 1e-10))
                        psnr = compute_psnr(ma_ct, mar_result.poly_ct, data_range)
                        ssim = compute_ssim(ma_ct, mar_result.poly_ct, data_range)
                        deg_disp = mu_to_display(ma_ct, window)
                        ax.imshow(deg_disp, cmap="gray")
                        ax.set_title(
                            f"mask #{col}\nPSNR={psnr:.1f}  SSIM={ssim:.4f}",
                            fontsize=9,
                        )
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes, fontsize=12)
                    ax.axis("off")
            else:
                for col, sev in enumerate(severities):
                    ax = axes[row, col + 1]
                    try:
                        sim = create_artifact_simulator(art_type, geo, phy, seed=42)
                        result = sim.simulate(hu_image, severity=sev)
                        clean_recon = result.poly_ct
                        degraded = result.artifact_results[0]["ma_CT"]

                        data_range = float(max(clean_recon.max() - clean_recon.min(), 1e-10))
                        psnr = compute_psnr(degraded, clean_recon, data_range)
                        ssim = compute_ssim(degraded, clean_recon, data_range)

                        deg_disp = mu_to_display(degraded, window)
                        ax.imshow(deg_disp, cmap="gray")
                        ax.set_title(
                            f"{sev}\nPSNR={psnr:.1f}  SSIM={ssim:.4f}",
                            fontsize=9,
                        )
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error:\n{str(e)[:80]}", ha="center", va="center",
                                fontsize=7, color="red", transform=ax.transAxes, wrap=True)
                        ax.set_title(f"{sev} (FAILED)", fontsize=9, color="red")
                    ax.axis("off")

            print(f"  {art_type} [{window}] done")

        fig.suptitle(
            f"All 10 CT Degradation Types — {case_label}  [{window} window]\n"
            f"Col 0: GT  |  Col 1-3: mild / moderate / severe (or mask #0/#1/#2 for metal)",
            fontsize=13, y=1.005,
        )
        fname = f"{case_label}_all_10_degradations_{window}.png"
        fig.savefig(str(out_dir / fname), dpi=120, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"Saved: {out_dir / fname}")

    # --- Difference maps (brain only, all 10 types) ---
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
    fig.subplots_adjust(hspace=0.30, wspace=0.05)

    for row, art_type in enumerate(artifact_types):
        if art_type == "metal":
            for col in range(3):
                ax = axes[row, col]
                if col < len(mar_result.mask_results):
                    mr = mar_result.mask_results[col]
                    diff = np.abs(mr["ma_CT"] - mar_result.poly_ct)
                    vmax = max(np.percentile(diff, 99), 0.005)
                    ax.imshow(diff, cmap="hot", vmin=0, vmax=vmax)
                    ax.set_title(f"metal / mask #{col}", fontsize=9)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(ARTIFACT_LABELS[art_type].split("\n")[0],
                                  fontsize=9, fontweight="bold", rotation=90, labelpad=8)
        else:
            for col, sev in enumerate(severities):
                ax = axes[row, col]
                try:
                    sim = create_artifact_simulator(art_type, geo, phy, seed=42)
                    result = sim.simulate(hu_image, severity=sev)
                    diff = np.abs(result.artifact_results[0]["ma_CT"] - result.poly_ct)
                    vmax = max(np.percentile(diff, 99), 0.005)
                    ax.imshow(diff, cmap="hot", vmin=0, vmax=vmax)
                    ax.set_title(f"{art_type} / {sev}", fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, str(e)[:60], ha="center", va="center",
                            fontsize=7, color="red", transform=ax.transAxes)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(ARTIFACT_LABELS[art_type].split("\n")[0],
                                  fontsize=9, fontweight="bold", rotation=90, labelpad=8)

    fig.suptitle(f"|Degraded − GT|  Difference Maps — {case_label}", fontsize=14, y=1.0)
    fname = f"{case_label}_diff_maps_10types.png"
    fig.savefig(str(out_dir / fname), dpi=120, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_dir / fname}")

    print("\nDone! All images saved to:", out_dir)


if __name__ == "__main__":
    main()
