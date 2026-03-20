#!/usr/bin/env python3
# ============================================================================
# 模块职责: 退化模拟脚本 — 对干净 CT 施加合成退化用于评估
# 参考: ProCT (https://github.com/Masaaki-75/proct)
#       PromptCT (https://github.com/shibaoshun/PromptCT)
# ============================================================================
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils import setup_logging
from src.degradations import DegradationSimulator, DegradationType
from src.io import read_ct, write_ct

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate CT degradation")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--type", choices=["noise", "blur", "low_resolution"], default="noise")
    parser.add_argument("--sigma", type=float, default=25.0, help="Noise sigma (for noise type)")
    args = parser.parse_args()

    setup_logging()
    simulator = DegradationSimulator()
    deg_type = DegradationType(args.type)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.*"))
    logger.info("Processing %d files with %s degradation", len(files), args.type)

    for f in files:
        try:
            image = read_ct(f)
            degraded = simulator.apply(image, deg_type, sigma=args.sigma)
            out_path = output_dir / f.relative_to(input_dir)
            write_ct(degraded, out_path)
        except Exception as e:
            logger.error("Failed on %s: %s", f, e)


if __name__ == "__main__":
    main()
