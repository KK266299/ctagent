#!/usr/bin/env python3
# ============================================================================
# 脚本职责: 生成逐 Slice 标签文件 (CSV + JSONL + stats)
#
# 用法:
#   PYTHONPATH=. python scripts/build_slice_labels.py \
#       --config configs/experiment/cq500_api_eval.yaml \
#       --strategy inherit_all \
#       --output-dir results/slice_labels
#
#   策略选项:
#     inherit_all — 有病灶 → 继承全部 case-level GT; 无病灶 → 全 0
#     bhx_aware   — 有病灶 → 出血类按 BHX label 映射; 非出血类继承 case-level
# ============================================================================
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_slice_labels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-slice labels for CQ500")
    parser.add_argument("--config", required=True, help="YAML config (cq500_api_eval.yaml)")
    parser.add_argument("--strategy", default="inherit_all",
                        choices=["inherit_all", "bhx_aware"],
                        help="Labeling strategy")
    parser.add_argument("--output-dir", default="results/slice_labels",
                        help="Output directory for label files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    from eval.bhx_loader import BHXAnnotations
    from eval.cq500_labels import CQ500Labels
    from eval.cq500_manifest import SOPIndex
    from eval.slice_labels import (
        SliceLabelGenerator,
        save_slice_labels_csv,
        save_slice_labels_jsonl,
        save_slice_label_stats,
    )

    labels = CQ500Labels(data_cfg["reads_csv"])
    logger.info("Loaded %d case-level labels", len(labels.gt))

    bhx_csv = data_cfg.get("bhx_csv")
    sop_index_path = data_cfg.get("sop_index")
    if not bhx_csv or not sop_index_path:
        logger.error("bhx_csv and sop_index must be configured in the YAML")
        return
    if not Path(bhx_csv).exists():
        logger.error("BHX CSV not found: %s", bhx_csv)
        return
    if not Path(sop_index_path).exists():
        logger.error("SOPIndex not found: %s — run scripts/build_sop_index.py first", sop_index_path)
        return

    bhx = BHXAnnotations(bhx_csv)
    sop_index = SOPIndex(sop_index_path)

    generator = SliceLabelGenerator(
        labels=labels,
        bhx=bhx,
        sop_index=sop_index,
        processed_dir=data_cfg["processed_dir"],
        strategy=args.strategy,
    )

    entries = generator.generate()

    out_dir = Path(args.output_dir)
    save_slice_labels_csv(entries, out_dir / "slice_labels.csv")
    save_slice_labels_jsonl(entries, out_dir / "slice_labels.jsonl")
    save_slice_label_stats(entries, out_dir / "stats.json")

    n_lesion = sum(1 for e in entries if e.lesion_present)
    n_bhx = sum(1 for e in entries if e.bhx_coverage)
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("  Total slices:          %d", len(entries))
    logger.info("  BHX-covered slices:    %d", n_bhx)
    logger.info("  Lesion-positive:       %d (%.1f%%)", n_lesion, 100 * n_lesion / max(len(entries), 1))
    logger.info("  Lesion-negative:       %d", len(entries) - n_lesion)
    logger.info("  Strategy:              %s", args.strategy)
    logger.info("  Output:                %s", out_dir)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
