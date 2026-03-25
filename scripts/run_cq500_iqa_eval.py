#!/usr/bin/env python
# ============================================================================
# 脚本职责: CQ500 IQA + 修复评测 CLI 入口
# 用法:
#   # 纯规则模式 (无需 API)
#   PYTHONPATH=. python scripts/run_cq500_iqa_eval.py \
#       --config configs/experiment/cq500_iqa_eval.yaml --planner-mode rule
#
#   # LLM 模式
#   source ~/network.sh
#   PYTHONPATH=. python scripts/run_cq500_iqa_eval.py \
#       --config configs/experiment/cq500_iqa_eval.yaml --planner-mode llm
#
#   # 两种对比
#   PYTHONPATH=. python scripts/run_cq500_iqa_eval.py \
#       --config configs/experiment/cq500_iqa_eval.yaml --planner-mode both
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
logger = logging.getLogger("cq500_iqa_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="CQ500 IQA + Restoration eval")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Override max_slices_per_case")
    parser.add_argument("--planner-mode", choices=["rule", "llm", "both"], default=None,
                        help="Override planner mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override LLM base URL")
    parser.add_argument("--replan-rounds", type=int, default=None,
                        help="LLM replan rounds (1=single, 2+=multi-round)")
    parser.add_argument("--test-only", type=str, default=None,
                        help="Path to train_test_split.json; only eval test cases (anti-leakage)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from eval.cq500_manifest import build_eval_manifest, SOPIndex
    from eval.cq500_iqa_restore_eval import (
        run_iqa_eval_batch,
        aggregate_iqa_results,
        compute_degradation_distribution,
        compute_tool_usage,
        save_iqa_outputs,
    )
    from src.degradations.detector import DegradationDetector
    from src.planner.rule_planner import RuleBasedPlanner
    from src.tools.mcp_style.restoration_tool import RestorationTool
    from src.tools.mcp_style.perception_tool import PerceptionTool

    # --- 触发工具注册 ---
    import src.tools.classical.denoise  # noqa: F401
    import src.tools.classical.sharpen  # noqa: F401
    import src.tools.classical.histogram  # noqa: F401
    import src.tools.classical.wavelet  # noqa: F401
    import src.tools.classical.median  # noqa: F401
    import src.tools.classical.deblur  # noqa: F401
    import src.tools.classical.enhance  # noqa: F401
    import src.tools.classical.inpaint  # noqa: F401
    import src.tools.classical.clip  # noqa: F401
    import src.tools.classical.mar  # noqa: F401
    import src.tools.classical.bm3d_denoise  # noqa: F401
    import src.tools.learned.mar_adapter  # noqa: F401
    import src.tools.learned.sr_adapter  # noqa: F401
    import src.tools.learned.dncnn_tool  # noqa: F401

    # --- 构建 manifest ---
    data_cfg = cfg.get("data", {})
    sop_index = None
    bhx_annotations = None

    bhx_csv = data_cfg.get("bhx_csv")
    sop_index_path = data_cfg.get("sop_index")
    if bhx_csv and sop_index_path:
        from pathlib import Path
        if Path(bhx_csv).exists() and Path(sop_index_path).exists():
            from eval.bhx_loader import BHXAnnotations
            bhx_annotations = BHXAnnotations(bhx_csv)
            sop_index = SOPIndex(sop_index_path)

    max_slices = args.max_slices or data_cfg.get("max_slices_per_case", 3)

    # --- 读取 test split (防数据泄漏) ---
    test_case_ids: set[str] | None = None
    if args.test_only:
        import json as _json
        from pathlib import Path as _P
        split_path = _P(args.test_only)
        if split_path.exists():
            with open(split_path) as _f:
                _split = _json.load(_f)
            test_case_ids = set(_split.get("test_cases", []))
            logger.info("Test-only mode: %d test cases from %s", len(test_case_ids), split_path)
        else:
            logger.warning("Split file not found: %s, evaluating all cases", split_path)

    cases = build_eval_manifest(
        processed_dir=data_cfg["processed_dir"],
        max_slices_per_case=max_slices,
        mask_idx=data_cfg.get("mask_idx", 0),
        sop_index=sop_index,
        bhx_annotations=bhx_annotations,
        label_case_ids=test_case_ids,
    )
    logger.info("Found %d evaluable cases", len(cases))

    if not cases:
        logger.error("No evaluable cases found.")
        sys.exit(1)

    if args.max_cases:
        cases = cases[:args.max_cases]
        logger.info("Limiting to %d cases", len(cases))

    # --- 初始化组件 ---
    detector_cfg = cfg.get("detector", {})
    detector = DegradationDetector(detector_cfg)

    rule_planner = RuleBasedPlanner(max_chain=cfg.get("planner", {}).get("max_chain", 3))
    perception_tool = PerceptionTool()
    restoration_tool = RestorationTool()

    # --- planner mode ---
    planner_mode = args.planner_mode or cfg.get("planner", {}).get("mode", "rule")

    agent_planner = None
    if planner_mode in ("llm", "both"):
        try:
            from llm.api_client import LLMConfig, create_client
            from llm.planner_caller import PlannerCaller
            from src.planner.agent_based import AgentBasedPlanner

            llm_cfg = cfg.get("llm", {})
            llm_config = LLMConfig(
                provider=llm_cfg.get("provider", "openai"),
                model=args.model or llm_cfg.get("model", "qwen/qwen-2.5-vl-72b-instruct"),
                base_url=args.base_url if args.base_url is not None else llm_cfg.get("base_url"),
                temperature=llm_cfg.get("temperature", 0.1),
                max_tokens=llm_cfg.get("max_tokens", 1024),
                timeout=llm_cfg.get("timeout", 120),
            )
            client = create_client(llm_config)
            planner_caller = PlannerCaller(llm_client=client)
            agent_planner = AgentBasedPlanner(
                planner_caller=planner_caller,
                detector_config=detector_cfg,
            )
            logger.info("LLM planner: %s / %s", llm_config.provider, llm_config.model)
        except Exception as e:
            logger.error("Failed to init LLM planner: %s", e)
            if planner_mode == "llm":
                sys.exit(1)
            planner_mode = "rule"

    planner_modes: list[str] = []
    if planner_mode == "both":
        planner_modes = ["rule", "llm"]
    else:
        planner_modes = [planner_mode]

    logger.info("Planner modes: %s", planner_modes)
    total_slices = sum(len(c.slices) for c in cases)
    logger.info("Total: %d cases, %d slices, %d evals",
                len(cases), total_slices, total_slices * len(planner_modes))

    # --- 运行评测 ---
    rate_limit = cfg.get("eval", {}).get("rate_limit_sec", 1.0)

    max_replan = args.replan_rounds or cfg.get("planner", {}).get("replan_rounds", 1)

    results = run_iqa_eval_batch(
        cases=cases,
        detector=detector,
        rule_planner=rule_planner,
        agent_planner=agent_planner,
        restoration_tool=restoration_tool,
        perception_tool=perception_tool,
        planner_modes=planner_modes,
        rate_limit_sec=rate_limit,
        max_replan_rounds=max_replan,
    )

    # --- 汇总与输出 ---
    summary = aggregate_iqa_results(results)
    deg_dist = compute_degradation_distribution(results)
    tool_usage = compute_tool_usage(results)

    output_dir = cfg.get("output", {}).get("dir", "results/cq500_iqa_eval")
    save_iqa_outputs(results, summary, deg_dist, tool_usage, output_dir)

    # --- 打印摘要 ---
    logger.info("=" * 60)
    logger.info("IQA EVALUATION SUMMARY")
    logger.info("=" * 60)
    for mode, m in summary.items():
        logger.info(
            "  [%s] PSNR: %.2f → %.2f (Δ%.2f)  SSIM: %.4f → %.4f (Δ%.4f)  "
            "success=%d/%d  latency=%.1fs",
            mode,
            m.get("psnr_before_mean", 0), m.get("psnr_after_mean", 0),
            m.get("psnr_improvement", 0),
            m.get("ssim_before_mean", 0), m.get("ssim_after_mean", 0),
            m.get("ssim_improvement", 0),
            m.get("n_restoration_success", 0), m.get("n_valid", 0),
            m.get("mean_latency_sec", 0),
        )
    logger.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
