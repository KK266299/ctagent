#!/usr/bin/env python3
# ============================================================================
# 模块职责: Toy Workflow CLI — 三模式对比验证
#   三种模式:
#     clean    → diagnosis   (上界: 无退化)
#     degraded → diagnosis   (下界: 有退化无修复)
#     degraded → restore → diagnosis (目标: 修复后效果)
#   输出对比表, 证明 restoration 对 diagnosis 的真实价值
# Usage:
#   PYTHONPATH=. python scripts/run_toy_workflow.py
#   PYTHONPATH=. python scripts/run_toy_workflow.py --num-cases 10 --seed 42
# 参考: JarvisIR — scripts/run_pipeline.py
#       4KAgent — evaluation scripts
# ============================================================================
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("toy_workflow")


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-Agent Toy Workflow (3-mode comparison)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of toy cases")
    parser.add_argument("--image-size", type=int, default=256, help="Phantom image size")
    parser.add_argument("--degradation", type=str, default="noise", choices=["noise", "blur", "low_resolution"])
    parser.add_argument("--sigma", type=float, default=0.08, help="Noise sigma (uniform)")
    parser.add_argument("--vary-sigma", action="store_true", default=True,
                        help="Vary sigma across cases (0.03 to 0.15)")
    parser.add_argument("--no-vary-sigma", dest="vary_sigma", action="store_false")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Save report JSON to path")
    args = parser.parse_args()

    num_cases = args.num_cases
    image_size = args.image_size
    degradation = args.degradation
    base_sigma = args.sigma
    vary_sigma = args.vary_sigma
    seed_start = args.seed

    if args.config:
        from src.utils.config import load_config
        cfg = load_config(args.config)
        toy_cfg = cfg.get("toy_data", {})
        num_cases = toy_cfg.get("num_cases", num_cases)
        image_size = toy_cfg.get("image_size", image_size)
        degradation = toy_cfg.get("degradation", degradation)
        base_sigma = toy_cfg.get("degradation_params", {}).get("sigma", base_sigma)
        seed_start = toy_cfg.get("seed_start", seed_start)
        vary_sigma = toy_cfg.get("vary_sigma", vary_sigma)

    if vary_sigma and degradation == "noise":
        sigma_min, sigma_max = 0.03, 0.15
        sigmas = [sigma_min + (sigma_max - sigma_min) * i / max(num_cases - 1, 1)
                  for i in range(num_cases)]
    else:
        sigmas = [base_sigma] * num_cases

    logger.info("=" * 65)
    logger.info("CT-Agent Toy Workflow — 3-mode comparison")
    logger.info("  cases=%d  size=%d  degradation=%s  sigma=%s  seed=%d",
                num_cases, image_size, degradation,
                f"vary[{sigmas[0]:.3f}-{sigmas[-1]:.3f}]" if vary_sigma else f"{base_sigma:.3f}",
                seed_start)
    logger.info("=" * 65)

    # ---- 1. 生成 toy cases ----
    from dataset.toy import generate_toy_case
    cases = []
    for i in range(num_cases):
        deg_params = {"sigma": sigmas[i]} if degradation == "noise" else {}
        case = generate_toy_case(
            size=image_size,
            degradation=degradation,
            degradation_params=deg_params,
            seed=seed_start + i,
        )
        cases.append(case)
        label = case["label"]
        contrast_info = ""
        if label.lesion_positions:
            contrasts = [p.get("contrast", "?") for p in label.lesion_positions]
            contrast_info = f", contrast={contrasts}"
        logger.info("Case %s (sigma=%.3f): lesion=%s count=%d side=%s%s",
                     case["case_id"], sigmas[i], label.lesion_present, label.lesion_count,
                     label.lesion_side, contrast_info)

    # ---- 2. 构建 pipeline ----
    from pipeline.single_pass import SinglePassPipeline
    from src.degradations.detector import DegradationDetector
    from src.planner.rule_planner import RuleBasedPlanner
    from src.degradations.types import DegradationType, Severity

    detector = DegradationDetector(config={
        "noise_thresholds": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    })
    toy_rules = {
        (DegradationType.NOISE, Severity.MILD): ["denoise_gaussian"],
        (DegradationType.NOISE, Severity.MODERATE): ["denoise_gaussian"],
        (DegradationType.NOISE, Severity.SEVERE): ["denoise_gaussian"],
        (DegradationType.BLUR, Severity.MILD): ["sharpen_usm"],
        (DegradationType.BLUR, Severity.SEVERE): ["sharpen_usm"],
    }
    planner = RuleBasedPlanner(rules=toy_rules)
    pipeline = SinglePassPipeline(detector=detector, planner=planner)

    # ---- 3. 三模式运行 ----
    clean_results = []
    degraded_results = []
    restored_results = []

    for case in cases:
        cid = case["case_id"]
        label = case["label"]

        r_clean = pipeline.diagnose_only(case["clean"], case_id=cid, mode="clean", label=label)
        clean_results.append(r_clean)

        r_degraded = pipeline.diagnose_only(case["degraded"], case_id=cid, mode="degraded", label=label)
        degraded_results.append(r_degraded)

        r_restored = pipeline.run_toy_case(case)
        restored_results.append(r_restored)

        logger.info("[%s] clean=%s | degraded=%s | restored=%s",
                     cid,
                     r_clean.diagnosis.get("prediction", "?"),
                     r_degraded.diagnosis.get("prediction", "?"),
                     r_restored.diagnosis.get("prediction", "?"))

    # ---- 4. 汇总 ----
    def accuracy(results):
        total = len(results)
        correct = sum(1 for r in results if r.diagnosis_correct)
        return correct, total

    c_ok, c_tot = accuracy(clean_results)
    d_ok, d_tot = accuracy(degraded_results)
    r_ok, r_tot = accuracy(restored_results)

    q_scores = [r.quality_verdict.get("score", 0) for r in restored_results if r.quality_verdict]
    s_scores = [r.safety_verdict.get("score", 0) for r in restored_results if r.safety_verdict]
    q_pass = sum(1 for r in restored_results if r.quality_verdict.get("passed"))
    s_pass = sum(1 for r in restored_results if r.safety_verdict.get("passed"))

    logger.info("")
    logger.info("=" * 65)
    logger.info("COMPARISON TABLE")
    logger.info("-" * 65)
    logger.info("%-20s  %10s  %10s  %10s", "", "clean", "degraded", "restored")
    logger.info("-" * 65)
    logger.info("%-20s  %7d/%-2d  %7d/%-2d  %7d/%-2d",
                "Diagnosis correct", c_ok, c_tot, d_ok, d_tot, r_ok, r_tot)
    logger.info("%-20s  %9.1f%%  %9.1f%%  %9.1f%%",
                "Diagnosis accuracy",
                100 * c_ok / max(c_tot, 1),
                100 * d_ok / max(d_tot, 1),
                100 * r_ok / max(r_tot, 1))
    logger.info("-" * 65)
    if q_scores:
        logger.info("%-20s  %10s  %10s  %7.3f",
                    "Quality score (avg)", "N/A", "N/A", sum(q_scores) / len(q_scores))
        logger.info("%-20s  %10s  %10s  %7d/%-2d",
                    "Quality pass", "N/A", "N/A", q_pass, r_tot)
    if s_scores:
        logger.info("%-20s  %10s  %10s  %7.3f",
                    "Safety score (avg)", "N/A", "N/A", sum(s_scores) / len(s_scores))
        logger.info("%-20s  %10s  %10s  %7d/%-2d",
                    "Safety pass", "N/A", "N/A", s_pass, r_tot)
    logger.info("=" * 65)

    delta = (r_ok - d_ok)
    if delta > 0:
        logger.info(">> Restoration improved diagnosis by +%d cases (%.1f%% -> %.1f%%)",
                     delta, 100 * d_ok / max(d_tot, 1), 100 * r_ok / max(r_tot, 1))
    elif delta == 0:
        logger.info(">> Restoration did not change diagnosis accuracy (%.1f%%)",
                     100 * d_ok / max(d_tot, 1))
    else:
        logger.info(">> WARNING: Restoration hurt diagnosis by %d cases", abs(delta))

    # ---- 5. 保存报告 ----
    summary = {
        "config": {
            "num_cases": num_cases, "image_size": image_size,
            "degradation": degradation, "degradation_params": deg_params,
            "seed_start": seed_start,
        },
        "comparison": {
            "clean": {"correct": c_ok, "total": c_tot, "accuracy": c_ok / max(c_tot, 1)},
            "degraded": {"correct": d_ok, "total": d_tot, "accuracy": d_ok / max(d_tot, 1)},
            "restored": {
                "correct": r_ok, "total": r_tot, "accuracy": r_ok / max(r_tot, 1),
                "quality_avg": sum(q_scores) / len(q_scores) if q_scores else 0,
                "quality_pass_rate": q_pass / max(r_tot, 1),
                "safety_avg": sum(s_scores) / len(s_scores) if s_scores else 0,
                "safety_pass_rate": s_pass / max(r_tot, 1),
            },
        },
        "cases": {
            "clean": [r.to_dict() for r in clean_results],
            "degraded": [r.to_dict() for r in degraded_results],
            "restored": [r.to_dict() for r in restored_results],
        },
    }

    if args.output:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
