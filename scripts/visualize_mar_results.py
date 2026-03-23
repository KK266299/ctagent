#!/usr/bin/env python
"""MAR 仿真结果可视化 — 生成 GT / ma_CT / LI_CT / BHC_CT 对比图。

对若干 slice 生成:
  1. 4-panel 对比图: GT | ma_CT (含伪影) | LI_CT (线性插值) | BHC_CT (BHC校正)
  2. 差异图: |ma_CT - GT|, |LI_CT - GT|
  3. 正弦图对比 + metal_trace 可视化

用法:
    PYTHONPATH=. python scripts/visualize_mar_results.py \
        --data-dir output/mar_cq500 \
        --output-dir try_output \
        --num-cases 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("visualize_mar")

MU_WINDOW = (0.0, 0.5)  # 线衰减系数显示窗
DIFF_WINDOW = (0.0, 0.15)


def load_case(gt_path: Path, mask_path: Path) -> dict[str, np.ndarray]:
    """加载一个 case 的所有数据。"""
    data = {}
    with h5py.File(str(gt_path), "r") as f:
        data["gt"] = f["image"][:].astype(np.float32)
        data["poly_CT"] = f["poly_CT"][:].astype(np.float32)
        data["poly_sinogram"] = f["poly_sinogram"][:].astype(np.float32)

    with h5py.File(str(mask_path), "r") as f:
        data["ma_CT"] = f["ma_CT"][:].astype(np.float32)
        data["LI_CT"] = f["LI_CT"][:].astype(np.float32)
        data["BHC_CT"] = f["BHC_CT"][:].astype(np.float32)
        data["ma_sinogram"] = f["ma_sinogram"][:].astype(np.float32)
        data["LI_sinogram"] = f["LI_sinogram"][:].astype(np.float32)
        data["metal_trace"] = f["metal_trace"][:].astype(np.float32)
    return data


def plot_4panel(data: dict, case_id: str, save_path: Path) -> None:
    """GT / ma_CT / LI_CT / BHC_CT 四图对比。"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["GT (clean)", "ma_CT (artifact)", "LI_CT (LI corrected)", "BHC_CT (BHC corrected)"]
    keys = ["gt", "ma_CT", "LI_CT", "BHC_CT"]

    for ax, title, key in zip(axes, titles, keys):
        im = ax.imshow(data[key], cmap="gray", vmin=MU_WINDOW[0], vmax=MU_WINDOW[1])
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"MAR Comparison — {case_id}", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_difference(data: dict, case_id: str, save_path: Path) -> None:
    """差异图: |ma - GT|, |LI - GT|, |BHC - GT|。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    diffs = [
        ("ma_CT", "|ma_CT − GT|"),
        ("LI_CT", "|LI_CT − GT|"),
        ("BHC_CT", "|BHC_CT − GT|"),
    ]
    gt = data["gt"]

    for ax, (key, title) in zip(axes, diffs):
        diff = np.abs(data[key] - gt)
        im = ax.imshow(diff, cmap="hot", vmin=DIFF_WINDOW[0], vmax=DIFF_WINDOW[1])
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Error Maps — {case_id}", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_sinograms(data: dict, case_id: str, save_path: Path) -> None:
    """正弦图对比 + metal_trace。"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    sino_keys = [
        ("poly_sinogram", "poly (clean)"),
        ("ma_sinogram", "ma (artifact)"),
        ("LI_sinogram", "LI corrected"),
        ("metal_trace", "Metal Trace"),
    ]

    for ax, (key, title) in zip(axes, sino_keys):
        if key == "metal_trace":
            ax.imshow(data[key], cmap="Reds", aspect="auto")
        else:
            vmax = np.percentile(data[key], 99)
            ax.imshow(data[key], cmap="gray", aspect="auto", vmin=0, vmax=max(vmax, 0.1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Sinograms — {case_id}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_zoomed(data: dict, case_id: str, save_path: Path) -> None:
    """中心区域放大对比 (128×128 crop)。"""
    h, w = data["gt"].shape
    cy, cx = h // 2, w // 2
    s = 64
    sl = (slice(cy - s, cy + s), slice(cx - s, cx + s))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["GT (zoom)", "ma_CT (zoom)", "LI_CT (zoom)", "BHC_CT (zoom)"]
    keys = ["gt", "ma_CT", "LI_CT", "BHC_CT"]

    for ax, title, key in zip(axes, titles, keys):
        ax.imshow(data[key][sl], cmap="gray", vmin=MU_WINDOW[0], vmax=MU_WINDOW[1])
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Center Zoom — {case_id}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="output/mar_cq500")
    parser.add_argument("--output-dir", default="try_output")
    parser.add_argument("--num-cases", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(data_dir.rglob("gt.h5"))[:args.num_cases]
    logger.info("Found %d cases in %s", len(gt_files), data_dir)

    for i, gt_path in enumerate(gt_files):
        parent = gt_path.parent
        mask_files = sorted(parent.glob("[0-9]*.h5"))
        if not mask_files:
            continue
        mask_path = mask_files[0]

        rel = parent.relative_to(data_dir)
        case_id = str(rel).replace("/", "_")
        logger.info("[%d/%d] %s", i + 1, len(gt_files), case_id)

        data = load_case(gt_path, mask_path)

        plot_4panel(data, case_id, output_dir / f"{case_id}_4panel.png")
        plot_difference(data, case_id, output_dir / f"{case_id}_diff.png")
        plot_sinograms(data, case_id, output_dir / f"{case_id}_sinogram.png")
        plot_zoomed(data, case_id, output_dir / f"{case_id}_zoom.png")

    logger.info("Saved %d × 4 plots to %s", len(gt_files), output_dir)


if __name__ == "__main__":
    main()
