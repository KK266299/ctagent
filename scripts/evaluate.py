#!/usr/bin/env python3
# ============================================================================
# 模块职责: 评估脚本 — 在配对数据集上批量评估修复效果
# 参考: MedQ-Bench (https://github.com/liujiyaoFDU/MedQ-Bench)
#       IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch)
# ============================================================================
from __future__ import annotations

import argparse
import logging

from src.utils import load_config, set_seed, setup_logging
from src.eval import PipelineEvaluator
from src.datasets import PairedCTDataset

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate restoration results")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--restored-dir", type=str, required=True)
    parser.add_argument("--reference-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/eval_report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging()
    set_seed(config.get("runtime", {}).get("seed", 42))

    metric_names = config.get("eval", {}).get("metrics", ["psnr", "ssim"])
    evaluator = PipelineEvaluator(metric_names=metric_names)

    dataset = PairedCTDataset(degraded_dir=args.restored_dir, clean_dir=args.reference_dir)
    logger.info("Evaluating %d samples", len(dataset))

    for i in range(len(dataset)):
        sample = dataset[i]
        evaluator.evaluate_single(sample["degraded"], sample["clean"], sample_id=str(i))

    summary = evaluator.summarize()
    logger.info("Summary: %s", summary)
    evaluator.save_report(args.output)


if __name__ == "__main__":
    main()
