#!/usr/bin/env python3
# ============================================================================
# 模块职责: Degraded Dataset 构建入口 — clean npy slices → degraded slices + manifest
# Usage:
#   PYTHONPATH=. python scripts/build_degraded_dataset.py
#   PYTHONPATH=. python scripts/build_degraded_dataset.py --config configs/data/degradation.yaml
#   PYTHONPATH=. python scripts/build_degraded_dataset.py --max-slices 100
# 参考: ADN — artifact synthesis 的可控退化思路
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
logger = logging.getLogger("build_degraded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build degraded 2D slice dataset from clean slices")
    parser.add_argument("--config", type=str, default="configs/data/degradation.yaml")
    parser.add_argument("--max-slices", type=int, default=None, help="Override max_slices")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    clean_manifest = cfg["input"]["clean_manifest"]
    max_slices = args.max_slices or cfg["input"].get("max_slices")
    out_cfg = cfg["output"]
    seed = cfg.get("seed", 42)

    from dataset.degradation_builder import DegradationConfig, build_degraded_dataset

    configs = []
    for d in cfg.get("degradations", []):
        configs.append(DegradationConfig(
            degradation_type=d["type"],
            severities=d.get("severities", [1, 2, 3]),
            params_override=d.get("params_override"),
        ))

    logger.info("=" * 70)
    logger.info("Build Degraded Dataset")
    logger.info("  clean_manifest: %s", clean_manifest)
    logger.info("  max_slices: %s", max_slices)
    logger.info("  degradations: %s", [(c.degradation_type, c.severities) for c in configs])
    logger.info("=" * 70)

    t0 = time.time()

    total = build_degraded_dataset(
        clean_manifest_path=clean_manifest,
        output_dir=out_cfg["degraded_dir"],
        degraded_manifest_path=out_cfg["manifest_path"],
        configs=configs,
        seed=seed,
        max_slices=max_slices,
    )

    elapsed = time.time() - t0
    logger.info("Done: %d degraded slices in %.1fs", total, elapsed)
    logger.info("Manifest: %s", out_cfg["manifest_path"])


if __name__ == "__main__":
    main()
