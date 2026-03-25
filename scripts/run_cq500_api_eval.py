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


def _setup_restore_fn(cfg: dict) -> None:
    """Initialize DnCNN restoration pipeline and register as restore_fn."""
    import numpy as np

    import src.tools.classical.clip  # noqa: F401
    import src.tools.learned.dncnn_tool  # noqa: F401
    from src.tools.registry import ToolRegistry

    clip_tool = ToolRegistry.create("clip_extreme")
    dncnn_tool = ToolRegistry.create("denoise_dncnn")

    def restore_fn(mu_image: np.ndarray) -> np.ndarray:
        result = clip_tool.run(mu_image)
        result = dncnn_tool.run(result.image)
        return result.image

    from eval.cq500_api_eval import set_restore_fn
    set_restore_fn(restore_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="CQ500 closed-API diagnosis eval")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Max number of cases to evaluate (for testing)")
    parser.add_argument("--input-types", nargs="+", default=None,
                        help="Override input types (e.g., clean degraded)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model (e.g., gpt-4o, openai/gpt-4o)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override LLM base URL (e.g., https://api.openai.com/v1)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override LLM temperature")
    parser.add_argument("--append", type=str, default=None,
                        help="Append results to existing predictions.jsonl and re-aggregate")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from eval.cq500_labels import CQ500Labels
    from eval.cq500_manifest import build_eval_manifest, SOPIndex
    from eval.cq500_api_eval import (
        run_batch_eval,
        aggregate_results,
        select_case_studies,
        save_outputs,
        CaseResult,
    )
    from llm.api_client import LLMConfig, create_client

    # --- 加载标签 ---
    labels_path = cfg["data"]["reads_csv"]
    labels = CQ500Labels(labels_path)
    logger.info("Loaded %d cases with GT labels", len(labels.gt))
    pos_counts = labels.positive_counts()
    logger.info("Positive counts: %s", pos_counts)

    # --- 加载 BHX 标注和 SOPIndex (可选) ---
    bhx_annotations = None
    sop_index = None
    manifest_cfg = cfg["data"]

    bhx_csv = manifest_cfg.get("bhx_csv")
    sop_index_path = manifest_cfg.get("sop_index")

    if bhx_csv and sop_index_path:
        from pathlib import Path
        if Path(bhx_csv).exists() and Path(sop_index_path).exists():
            from eval.bhx_loader import BHXAnnotations
            bhx_annotations = BHXAnnotations(bhx_csv)
            sop_index = SOPIndex(sop_index_path)
            logger.info(
                "Lesion-aware slice selection enabled: BHX=%s, SOPIndex=%s",
                bhx_csv, sop_index_path,
            )
        else:
            if not Path(bhx_csv).exists():
                logger.warning("BHX CSV not found: %s", bhx_csv)
            if not Path(sop_index_path).exists():
                logger.warning(
                    "SOPIndex not found: %s — run scripts/build_sop_index.py first",
                    sop_index_path,
                )
    else:
        logger.info("BHX not configured — using middle_uniform slice selection")

    # --- 构建 manifest ---
    cases = build_eval_manifest(
        processed_dir=manifest_cfg["processed_dir"],
        label_case_ids=set(labels.case_ids()),
        max_slices_per_case=manifest_cfg.get("max_slices_per_case", 5),
        mask_idx=manifest_cfg.get("mask_idx", 0),
        restored_dir=manifest_cfg.get("restored_dir"),
        sop_index=sop_index,
        bhx_annotations=bhx_annotations,
    )

    if bhx_annotations:
        n_lesion = sum(1 for c in cases if c.n_lesion_slices > 0)
        methods = {}
        for c in cases:
            methods[c.selection_method] = methods.get(c.selection_method, 0) + 1
        logger.info(
            "Slice selection stats: %d/%d cases with lesion slices, methods=%s",
            n_lesion, len(cases), methods,
        )
    logger.info("Found %d evaluable cases", len(cases))

    if not cases:
        logger.error("No evaluable cases found. Check processed_dir and labels alignment.")
        sys.exit(1)

    if args.max_cases:
        cases = cases[:args.max_cases]
        logger.info("Limiting to %d cases", len(cases))

    # --- 创建 LLM client (命令行参数优先于配置文件) ---
    llm_cfg_dict = cfg.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_cfg_dict.get("provider", "openai"),
        model=args.model or llm_cfg_dict.get("model", "gpt-4o"),
        base_url=args.base_url if args.base_url is not None else llm_cfg_dict.get("base_url"),
        temperature=args.temperature if args.temperature is not None else llm_cfg_dict.get("temperature", 0.1),
        max_tokens=llm_cfg_dict.get("max_tokens", 1024),
        timeout=llm_cfg_dict.get("timeout", 120),
    )
    client = create_client(llm_config)
    logger.info("LLM: %s / %s (base_url=%s)", llm_config.provider, llm_config.model, llm_config.base_url)

    # --- 确定评测路数 ---
    input_types = args.input_types or cfg.get("eval", {}).get("input_types", ["clean", "degraded"])
    logger.info("Input types: %s", input_types)

    # --- 如果包含 restored，初始化修复管线 ---
    if "restored" in input_types:
        from eval.cq500_api_eval import set_restore_fn
        _setup_restore_fn(cfg)
        logger.info("Restoration pipeline initialized for 'restored' input type")

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

    # --- 如果追加模式，合并已有结果 ---
    output_dir = cfg.get("output", {}).get("dir", "results/cq500_api_eval")

    if args.append:
        from pathlib import Path
        import json as _json
        existing_path = Path(args.append)
        if existing_path.exists():
            from llm.response_parser import CQ500DiagnosisResult
            existing_results = []
            with open(existing_path) as _f:
                for line in _f:
                    rec = _json.loads(line)
                    cr = CaseResult(
                        case_id=rec["case_id"],
                        input_type=rec["input_type"],
                        gt_labels=rec["gt"],
                        pred=CQ500DiagnosisResult(
                            predictions=rec.get("predictions", {}),
                            confidence=rec.get("confidence", {}),
                            reasoning=rec.get("reasoning", ""),
                            parse_success=rec.get("parse_success", True),
                        ),
                        raw_prompt_text=rec.get("raw_prompt", ""),
                        raw_response_text=rec.get("raw_response", ""),
                        api_latency_sec=rec.get("api_latency_sec", 0),
                        error=rec.get("error"),
                        usage=rec.get("usage", {}),
                    )
                    existing_results.append(cr)
            logger.info("Loaded %d existing results from %s", len(existing_results), existing_path)
            results = existing_results + results
            all_types = list(dict.fromkeys(r.input_type for r in results))
        else:
            logger.warning("Append file not found: %s", existing_path)
            all_types = input_types
    else:
        all_types = input_types

    # --- 汇总与输出 ---
    summary = aggregate_results(results, all_types)
    case_studies = select_case_studies(results)

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
