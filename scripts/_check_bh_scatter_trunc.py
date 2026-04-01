#!/usr/bin/env python
"""Beam Hardening / Scatter / Truncation：正弦图 + |正弦图差| + CT（脑窗/硬膜下窗）。

正弦图为 BHC 后 sinogram 域（与 poly_sinogram / ma_sinogram 一致），形状 (num_views, num_bins)。

输出:
  {output_dir}/check_bh_scatter_trunc_sinogram.png  — 三类总览
  {output_dir}/check_beam_hardening_sino_ct.png
  {output_dir}/check_scatter_sino_ct.png
  {output_dir}/check_truncation_sino_ct.png

用法:
  PYTHONPATH=. python scripts/_check_bh_scatter_trunc.py --gpu 7 \\
      --output-dir try_output/degradation_review
"""
from __future__ import annotations

import argparse
import os
import pathlib

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MU_WATER = 0.192


def mu_to_display(img: np.ndarray, wl: float = 40, ww: float = 80) -> np.ndarray:
    hu_lo = wl - ww / 2.0
    hu_hi = wl + ww / 2.0
    lo = MU_WATER * (1.0 + hu_lo / 1000.0)
    hi = MU_WATER * (1.0 + hu_hi / 1000.0)
    return np.clip((img - lo) / max(hi - lo, 1e-10), 0.0, 1.0)


def sino_display(sino: np.ndarray, pct_lo: float = 1, pct_hi: float = 99) -> np.ndarray:
    lo = np.percentile(sino, pct_lo)
    hi = np.percentile(sino, pct_hi)
    return np.clip((sino - lo) / max(hi - lo, 1e-10), 0.0, 1.0)


def find_middle_mu_h5(processed_dir: pathlib.Path) -> tuple[np.ndarray, str]:
    for cd in sorted(processed_dir.iterdir()):
        if not cd.is_dir():
            continue
        for sd in sorted(cd.iterdir()):
            slices = sorted([d for d in sd.iterdir() if d.is_dir()])
            if len(slices) < 5:
                continue
            gt_path = slices[len(slices) // 2] / "gt.h5"
            if gt_path.exists():
                with h5py.File(str(gt_path), "r") as f:
                    mu_image = f["image"][:].astype(np.float64)
                rel = gt_path.parent.relative_to(processed_dir)
                return mu_image, str(rel).replace("/", "_")
    raise FileNotFoundError(f"No gt.h5 under {processed_dir}")


def plot_panels_4x4(
    axes: np.ndarray,
    results: dict[str, object],
    gt_sino: np.ndarray,
    gt_ct: np.ndarray,
    art_title: str,
    sevs: list[str],
    compute_psnr,
    compute_ssim,
) -> None:
    """Fill a 4×4 grid: sino, |sino diff|, CT brain, CT subdural."""
    nv, nb = gt_sino.shape
    sino_caption = f"BHC sinogram ({nv}×{nb}, view×bin)"

    ax = axes[0, 0]
    ax.imshow(sino_display(gt_sino), cmap="gray", aspect="auto", origin="upper")
    ax.set_title(f"GT\n{sino_caption}", fontsize=8)
    ax.set_ylabel(
        f"{art_title}\nSinogram\n(view ↓, bin →)",
        fontsize=9,
        fontweight="bold",
        rotation=0,
        labelpad=72,
        va="center",
    )
    ax.set_xlabel("detector bin", fontsize=7)
    ax.tick_params(labelsize=5)

    for col, sev in enumerate(sevs):
        ax = axes[0, col + 1]
        deg_sino = results[sev].artifact_results[0]["ma_sinogram"]
        ax.imshow(sino_display(deg_sino), cmap="gray", aspect="auto", origin="upper")
        ax.set_title(f"{sev}\n{sino_caption}", fontsize=8)
        ax.set_xlabel("detector bin", fontsize=7)
        ax.tick_params(labelsize=5)

    ax_lbl = axes[1, 0]
    ax_lbl.axis("off")
    ax_lbl.text(
        0.5,
        0.5,
        f"{art_title}\n|Δsino|",
        ha="center",
        va="center",
        transform=ax_lbl.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    for col, sev in enumerate(sevs):
        ax = axes[1, col + 1]
        deg_sino = results[sev].artifact_results[0]["ma_sinogram"]
        diff_sino = np.abs(deg_sino.astype(np.float64) - gt_sino.astype(np.float64))
        vmax = max(np.percentile(diff_sino, 99), 1e-6)
        ax.imshow(diff_sino, cmap="hot", aspect="auto", origin="upper", vmin=0, vmax=vmax)
        ax.set_title(f"{sev} |GT−deg|\nmax={diff_sino.max():.4f}", fontsize=8)
        ax.set_xlabel("detector bin", fontsize=7)
        ax.tick_params(labelsize=5)

    ax = axes[2, 0]
    ax.imshow(mu_to_display(gt_ct, 40, 80), cmap="gray")
    ax.set_title("GT brain\nW40/L80 (μ)", fontsize=8)
    ax.set_ylabel(f"{art_title}\nCT brain", fontsize=10, fontweight="bold", rotation=0, labelpad=88, va="center")
    ax.axis("off")

    for col, sev in enumerate(sevs):
        ax = axes[2, col + 1]
        deg_ct = results[sev].artifact_results[0]["ma_CT"]
        dr = float(max(gt_ct.max() - gt_ct.min(), 1e-10))
        p = compute_psnr(deg_ct, gt_ct, dr)
        s = compute_ssim(deg_ct, gt_ct, dr)
        ax.imshow(mu_to_display(deg_ct, 40, 80), cmap="gray")
        ax.set_title(f"{sev}\nPSNR={p:.1f} SSIM={s:.4f}", fontsize=8)
        ax.axis("off")

    ax = axes[3, 0]
    ax.imshow(mu_to_display(gt_ct, 75, 215), cmap="gray")
    ax.set_title("GT subdural\nW75/L215 (μ)", fontsize=8)
    ax.set_ylabel(f"{art_title}\nCT subdural", fontsize=10, fontweight="bold", rotation=0, labelpad=88, va="center")
    ax.axis("off")

    for col, sev in enumerate(sevs):
        ax = axes[3, col + 1]
        deg_ct = results[sev].artifact_results[0]["ma_CT"]
        dr = float(max(gt_ct.max() - gt_ct.min(), 1e-10))
        p = compute_psnr(deg_ct, gt_ct, dr)
        s = compute_ssim(deg_ct, gt_ct, dr)
        ax.imshow(mu_to_display(deg_ct, 75, 215), cmap="gray")
        ax.set_title(f"{sev}\nPSNR={p:.1f} SSIM={s:.4f}", fontsize=8)
        ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--output-dir", default="try_output/degradation_review")
    parser.add_argument("--processed-dir", default="/home/liuxinyao/data/cq500_processed")
    parser.add_argument("--mat-dir", default="data/mar_physics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from dataset.mar.ct_artifact_simulator import create_artifact_simulator
    from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig
    from dataset.mar.physics_params import PhysicsConfig, PhysicsParams
    from src.iqa.metrics import compute_psnr, compute_ssim

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = pathlib.Path(args.processed_dir)

    geo = CTGeometry(CTGeometryConfig(image_size=416, num_angles=640, num_detectors=641, impl="astra_cuda"))
    phy = PhysicsParams(PhysicsConfig(mat_dir=args.mat_dir))
    phy.load()

    mu_image, case_label = find_middle_mu_h5(processed)
    hu_image = (mu_image / MU_WATER - 1.0) * 1000.0
    print(f"Slice case: {case_label}")

    art_types = ["beam_hardening", "scatter", "truncation"]
    art_labels = {
        "beam_hardening": "Beam Hardening",
        "scatter": "Scatter",
        "truncation": "Truncation",
    }
    sevs = ["mild", "moderate", "severe"]

    all_results: dict[str, dict[str, object]] = {}
    for art_type in art_types:
        results = {}
        for sev in sevs:
            sim = create_artifact_simulator(art_type, geo, phy, seed=args.seed)
            results[sev] = sim.simulate(hu_image, severity=sev)
        all_results[art_type] = results
        ref = results["mild"]
        gt_sino = ref.poly_sinogram
        gt_ct = ref.poly_ct

        fig_s, ax_s = plt.subplots(4, 4, figsize=(16, 14))
        fig_s.subplots_adjust(hspace=0.45, wspace=0.12)
        plot_panels_4x4(ax_s, results, gt_sino, gt_ct, art_labels[art_type], sevs, compute_psnr, compute_ssim)
        fig_s.suptitle(
            f"{art_labels[art_type]} — sinogram + CT  |  case={case_label}",
            fontsize=12,
            y=1.01,
        )
        single_name = f"check_{art_type}_sino_ct.png"
        fig_s.savefig(str(out_dir / single_name), dpi=130, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig_s)
        print(f"Saved: {out_dir / single_name}")

    n_blocks = len(art_types)
    fig, axes = plt.subplots(n_blocks * 4, 4, figsize=(18, 3.0 * n_blocks * 4))
    fig.subplots_adjust(hspace=0.5, wspace=0.12)

    for art_idx, art_type in enumerate(art_types):
        results = all_results[art_type]
        ref = results["mild"]
        gt_sino = ref.poly_sinogram
        gt_ct = ref.poly_ct
        base = art_idx * 4
        plot_panels_4x4(
            axes[base : base + 4, :],
            results,
            gt_sino,
            gt_ct,
            art_labels[art_type],
            sevs,
            compute_psnr,
            compute_ssim,
        )
        print(f"  panel {art_type} in overview done")

    fig.suptitle(
        f"Beam Hardening / Scatter / Truncation — sinogram + CT  |  {case_label}\n"
        "Per block: row1 sinogram (BHC domain) → row2 |Δsino| → row3 CT brain → row4 CT subdural",
        fontsize=12,
        y=1.002,
    )
    overview = "check_bh_scatter_trunc_sinogram.png"
    fig.savefig(str(out_dir / overview), dpi=110, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_dir / overview}")

    print("\n=== Value statistics (sinogram / CT) ===")
    for art_type in art_types:
        print(f"\n--- {art_labels[art_type]} ---")
        for sev in sevs:
            sim = create_artifact_simulator(art_type, geo, phy, seed=args.seed)
            r = sim.simulate(hu_image, severity=sev)
            gt_s = r.poly_sinogram
            gt_c = r.poly_ct
            d_s = r.artifact_results[0]["ma_sinogram"]
            d_c = r.artifact_results[0]["ma_CT"]
            params = r.artifact_results[0].get("params", {})
            sino_diff = np.abs(d_s.astype(float) - gt_s.astype(float))
            ct_diff = np.abs(d_c.astype(float) - gt_c.astype(float))
            print(f"  {sev:10s}: params={params}")
            print(
                f"    sino: gt[{gt_s.min():.4f},{gt_s.max():.4f}]  "
                f"deg[{d_s.min():.4f},{d_s.max():.4f}]  "
                f"|Δ|: mean={sino_diff.mean():.5f} max={sino_diff.max():.5f}"
            )
            print(
                f"    ct:   gt[{gt_c.min():.4f},{gt_c.max():.4f}]  "
                f"deg[{d_c.min():.4f},{d_c.max():.4f}]  "
                f"|Δ|: mean={ct_diff.mean():.6f} max={ct_diff.max():.5f}"
            )

    print("\nDone")


if __name__ == "__main__":
    main()
