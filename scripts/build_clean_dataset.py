#!/usr/bin/env python3
# ============================================================================
# 模块职责: Clean Dataset 构建入口 — DICOM → clean 2D npy slices + manifest
# Usage:
#   PYTHONPATH=. python scripts/build_clean_dataset.py
#   PYTHONPATH=. python scripts/build_clean_dataset.py --config configs/data/dicom_dataset.yaml
#   PYTHONPATH=. python scripts/build_clean_dataset.py --max-patients 2 --max-series 5
# 参考: ADN — prepare_spineweb.py
# ============================================================================
from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_clean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean 2D slice dataset from DICOM")
    parser.add_argument("--config", type=str, default="configs/data/dicom_dataset.yaml")
    parser.add_argument("--root-dir", type=str, default=None, help="Override input root_dir")
    parser.add_argument("--max-patients", type=int, default=None, help="Override max_patients")
    parser.add_argument("--max-series", type=int, default=None, help="Override max_series")
    parser.add_argument("--window-center", type=float, default=None)
    parser.add_argument("--window-width", type=float, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root_dir = args.root_dir or cfg["input"]["root_dir"]
    max_patients = args.max_patients or cfg["input"].get("max_patients")
    sf = cfg.get("filter", {})
    wc = args.window_center or cfg.get("window", {}).get("center")
    ww = args.window_width or cfg.get("window", {}).get("width")
    out_cfg = cfg["output"]
    max_series = args.max_series or out_cfg.get("max_series")

    logger.info("=" * 70)
    logger.info("Build Clean Dataset")
    logger.info("  root_dir: %s", root_dir)
    logger.info("  max_patients: %s  max_series: %s", max_patients, max_series)
    logger.info("  window: center=%s  width=%s", wc, ww)
    logger.info("  filter: %s", sf)
    logger.info("=" * 70)

    t0 = time.time()

    from dataset.dicom_scanner import scan_dicom_directory
    series_list = scan_dicom_directory(
        root_dir=root_dir,
        max_patients=max_patients,
        series_filter=sf,
    )

    if not series_list:
        logger.error("No series found! Check root_dir and filter settings.")
        return

    from dataset.ct_slice_exporter import export_clean_slices
    total = export_clean_slices(
        series_list=series_list,
        output_dir=out_cfg["clean_dir"],
        manifest_path=out_cfg["manifest_path"],
        window_center=wc,
        window_width=ww,
        max_series=max_series,
    )

    elapsed = time.time() - t0
    logger.info("Done: %d slices in %.1fs", total, elapsed)
    logger.info("Manifest: %s", out_cfg["manifest_path"])


if __name__ == "__main__":
    main()
