#!/usr/bin/env python3
# ============================================================================
# 模块职责: 主 pipeline 脚本 — 端到端运行: 退化检测 → 规划 → 修复 → 评估
# 参考: JarvisIR (https://github.com/LYL1015/JarvisIR) — pipeline runner
#       MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# ============================================================================
from __future__ import annotations

import argparse
import logging

from src.utils import load_config, set_seed, setup_logging
from src.degradations import DegradationDetector
from src.planner import RuleBasedPlanner
from src.executor import Executor

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-Agent-Frontdoor Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    parser.add_argument("--input", type=str, help="Input CT image path")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("runtime", {}).get("log_level", "INFO"))
    set_seed(config.get("runtime", {}).get("seed", 42))

    logger.info("Loaded config: %s", args.config)

    # 1. 退化检测
    detector = DegradationDetector()
    # 2. 规划
    planner = RuleBasedPlanner()
    # 3. 执行器
    executor = Executor()

    if args.input:
        from src.io import read_ct
        image = read_ct(args.input)
        logger.info("Input image shape: %s", image.shape)

        report = detector.detect(image)
        logger.info("Degradation report: %s", report)

        plan = planner.plan(report)
        logger.info("Plan: %s", plan.tool_names())

        results = executor.execute(plan, image)
        logger.info("Execution complete: %d steps", len(results))

        if results:
            from src.io import write_ct
            from pathlib import Path
            out_path = Path(args.output_dir) / "restored.png"
            write_ct(results[-1].image, out_path)
            logger.info("Saved to %s", out_path)
    else:
        logger.info("No input specified. Use --input to provide a CT image.")


if __name__ == "__main__":
    main()
