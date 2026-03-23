#!/usr/bin/env python
# ============================================================================
# 脚本职责: CQ500 闭源 API 诊断分类评测 CLI 入口
# 用法:
#   PYTHONPATH=. python scripts/run_cq500_api_eval.py \
#       --config configs/experiment/cq500_api_eval.yaml --max-cases 3
# ============================================================================
from __future__ import annotations

import argparse
import logging
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cq500_api_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="CQ500 closed-API diagnosis eval")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Max number of cases to evaluate (for testing)")
    parser.add_argument("--input-types", nargs="+", default=None,
                        help="Override input types (e.g., clean degraded)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from eval.cq500_labels import CQ500Labels
    from eval.cq500_manifest import build_eval_manifest
    from eval.cq500_api_eval import (
        run_batch_eval,
        aggregate_results,
        select_case_studies,
        save_outputs,
    )
    from llm.api_client import LLMConfig, create_client

    # --- 加载标签 ---
    labels_path = cfg["data"]["reads_csv"]
    labels = CQ500Labels(labels_path)
    logger.info("Loaded %d cases with GT labels", len(labels.gt))
    pos_counts = labels.positive_counts()
    logger.info("Positive counts: %s", pos_counts)

    # --- 构建 manifest ---
    manifest_cfg = cfg["data"]
    cases = build_eval_manifest(
        processed_dir=manifest_cfg["processed_dir"],
        label_case_ids=set(labels.case_ids()),
        max_slices_per_case=manifest_cfg.get("max_slices_per_case", 5),
        mask_idx=manifest_cfg.get("mask_idx", 0),
        restored_dir=manifest_cfg.get("restored_dir"),
    )
    logger.info("Found %d evaluable cases", len(cases))

    if not cases:
        logger.error("No evaluable cases found. Check processed_dir and labels alignment.")
        sys.exit(1)

    if args.max_cases:
        cases = cases[:args.max_cases]
        logger.info("Limiting to %d cases", len(cases))

    # --- 创建 LLM client ---
    llm_cfg_dict = cfg.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_cfg_dict.get("provider", "openai"),
        model=llm_cfg_dict.get("model", "gpt-4o"),
        base_url=llm_cfg_dict.get("base_url"),
        temperature=llm_cfg_dict.get("temperature", 0.1),
        max_tokens=llm_cfg_dict.get("max_tokens", 1024),
        timeout=llm_cfg_dict.get("timeout", 120),
    )
    client = create_client(llm_config)
    logger.info("LLM: %s / %s", llm_config.provider, llm_config.model)

    # --- 确定评测路数 ---
    input_types = args.input_types or cfg.get("eval", {}).get("input_types", ["clean", "degraded"])
    logger.info("Input types: %s", input_types)

    # --- 运行评测 ---
    windowing = cfg.get("windowing", {})
    rate_limit = cfg.get("eval", {}).get("rate_limit_sec", 1.0)

    results = run_batch_eval(
        llm_client=client,
        cases=cases,
        labels=labels,
        input_types=input_types,
        mu_water=windowing.get("mu_water", 0.192),
        window_center=windowing.get("window_center", 40.0),
        window_width=windowing.get("window_width", 80.0),
        rate_limit_sec=rate_limit,
    )

    # --- 汇总与输出 ---
    summary = aggregate_results(results, input_types)
    case_studies = select_case_studies(results)

    output_dir = cfg.get("output", {}).get("dir", "results/cq500_api_eval")
    save_outputs(results, summary, case_studies, output_dir)

    # --- 打印摘要 ---
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for itype, m in summary.get("per_input_type", {}).items():
        logger.info(
            "  [%s] accuracy=%.4f  macro_F1=%.4f  micro_F1=%.4f  "
            "auroc=%s  valid=%d  failed=%d  latency=%.1fs",
            itype,
            m.get("mean_accuracy", 0),
            m.get("macro_f1", 0),
            m.get("micro_f1", 0),
            m.get("mean_auroc", "N/A"),
            m.get("n_valid", 0),
            m.get("n_failed", 0),
            m.get("mean_latency_sec", 0),
        )

    dr = summary.get("drop_recovery")
    if dr:
        logger.info("  Degraded drop: %.2f%%", dr.get("degraded_drop_pct", 0))
        if dr.get("restored_recovery_pct") is not None:
            logger.info("  Restored recovery: %.2f%%", dr["restored_recovery_pct"])

    logger.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
