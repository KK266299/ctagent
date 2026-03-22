#!/usr/bin/env python3
# ============================================================================
# 模块职责: Closed-Loop Demo CLI — single-pass vs closed-loop 对比
#   在相同的 toy cases 上同时运行:
#     1. single-pass pipeline (1 轮, 无 replan)
#     2. closed-loop pipeline (最多 N 轮, judge FAIL 触发 replan)
#   输出对比表: diagnosis accuracy, quality/safety, iterations, replan count
# Usage:
#   PYTHONPATH=. python scripts/run_closed_loop_demo.py
#   PYTHONPATH=. python scripts/run_closed_loop_demo.py --num-cases 10 --max-iter 3
# 参考: 4KAgent — ablation: open-loop vs closed-loop
# ============================================================================
from __future__ import annotations

import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("closed_loop_demo")


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-Agent: Single-Pass vs Closed-Loop")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=3, help="Max closed-loop iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Save report JSON")
    args = parser.parse_args()

    num_cases = args.num_cases
    seed_start = args.seed
    max_iter = args.max_iter

    sigma_min, sigma_max = 0.03, 0.15
    sigmas = [sigma_min + (sigma_max - sigma_min) * i / max(num_cases - 1, 1)
              for i in range(num_cases)]

    logger.info("=" * 70)
    logger.info("CT-Agent: Single-Pass vs Closed-Loop (max_iter=%d)", max_iter)
    logger.info("  cases=%d  sigma=vary[%.3f-%.3f]  seed=%d", num_cases, sigmas[0], sigmas[-1], seed_start)
    logger.info("=" * 70)

    # ---- 1. Generate cases ----
    from dataset.toy import generate_toy_case
    cases = []
    for i in range(num_cases):
        case = generate_toy_case(
            size=args.image_size,
            degradation="noise",
            degradation_params={"sigma": sigmas[i]},
            seed=seed_start + i,
        )
        cases.append(case)

    # ---- 2. Build pipelines (shared components) ----
    from src.degradations.detector import DegradationDetector
    from src.planner.rule_planner import RuleBasedPlanner
    from src.degradations.types import DegradationType, Severity
    from pipeline.single_pass import SinglePassPipeline
    from pipeline.agent_loop import ClosedLoopPipeline

    detector = DegradationDetector(config={
        "noise_thresholds": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    })
    toy_rules = {
        (DegradationType.NOISE, Severity.MILD): ["denoise_gaussian"],
        (DegradationType.NOISE, Severity.MODERATE): ["denoise_gaussian"],
        (DegradationType.NOISE, Severity.SEVERE): ["denoise_gaussian"],
    }
    planner = RuleBasedPlanner(rules=toy_rules)

    sp_pipeline = SinglePassPipeline(detector=detector, planner=planner)
    cl_pipeline = ClosedLoopPipeline(detector=detector, planner=planner, max_iterations=max_iter)

    # ---- 3. Run both ----
    sp_results = []
    cl_results = []

    for idx, case in enumerate(cases):
        cid = case["case_id"]
        sigma = sigmas[idx]
        label = case["label"]

        logger.info("-" * 50)
        logger.info("[%s] sigma=%.3f  lesion=%s count=%d", cid, sigma, label.lesion_present, label.lesion_count)

        sp_r = sp_pipeline.run_toy_case(case)
        sp_results.append(sp_r)

        cl_r = cl_pipeline.run_toy_case(case)
        cl_results.append(cl_r)

        sp_q = sp_r.quality_verdict.get("score", 0)
        sp_s = sp_r.safety_verdict.get("score", 0)
        cl_q = cl_r.final_quality.get("score", 0)
        cl_s = cl_r.final_safety.get("score", 0)

        logger.info("  single-pass: q=%.3f s=%.3f diag=%s correct=%s",
                     sp_q, sp_s, sp_r.diagnosis.get("prediction", "?"), sp_r.diagnosis_correct)
        logger.info("  closed-loop: q=%.3f s=%.3f diag=%s correct=%s (iter=%d, replan=%d)",
                     cl_q, cl_s, cl_r.diagnosis.get("prediction", "?"), cl_r.diagnosis_correct,
                     cl_r.total_iterations, cl_r.replan_count)

    # ---- 4. Summary ----
    def acc(results):
        total = len(results)
        correct = sum(1 for r in results if r.diagnosis_correct)
        return correct, total

    sp_ok, sp_tot = acc(sp_results)
    cl_ok, cl_tot = acc(cl_results)

    sp_q_scores = [r.quality_verdict.get("score", 0) for r in sp_results if r.quality_verdict]
    sp_s_scores = [r.safety_verdict.get("score", 0) for r in sp_results if r.safety_verdict]
    sp_q_pass = sum(1 for r in sp_results if r.quality_verdict.get("passed"))
    sp_s_pass = sum(1 for r in sp_results if r.safety_verdict.get("passed"))

    cl_q_scores = [r.final_quality.get("score", 0) for r in cl_results]
    cl_s_scores = [r.final_safety.get("score", 0) for r in cl_results]
    cl_q_pass = sum(1 for r in cl_results if r.final_quality.get("passed"))
    cl_s_pass = sum(1 for r in cl_results if r.final_safety.get("passed"))

    avg_iter = sum(r.total_iterations for r in cl_results) / max(len(cl_results), 1)
    total_replans = sum(r.replan_count for r in cl_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: Single-Pass vs Closed-Loop")
    logger.info("-" * 70)
    logger.info("%-25s  %12s  %12s", "", "single-pass", "closed-loop")
    logger.info("-" * 70)
    logger.info("%-25s  %9d/%-2d  %9d/%-2d",
                "Diagnosis correct", sp_ok, sp_tot, cl_ok, cl_tot)
    logger.info("%-25s  %11.1f%%  %11.1f%%",
                "Diagnosis accuracy",
                100 * sp_ok / max(sp_tot, 1),
                100 * cl_ok / max(cl_tot, 1))
    logger.info("-" * 70)
    logger.info("%-25s  %11.3f  %11.3f",
                "Quality score (avg)",
                sum(sp_q_scores) / max(len(sp_q_scores), 1),
                sum(cl_q_scores) / max(len(cl_q_scores), 1))
    logger.info("%-25s  %9d/%-2d  %9d/%-2d",
                "Quality pass", sp_q_pass, sp_tot, cl_q_pass, cl_tot)
    logger.info("%-25s  %11.3f  %11.3f",
                "Safety score (avg)",
                sum(sp_s_scores) / max(len(sp_s_scores), 1),
                sum(cl_s_scores) / max(len(cl_s_scores), 1))
    logger.info("%-25s  %9d/%-2d  %9d/%-2d",
                "Safety pass", sp_s_pass, sp_tot, cl_s_pass, cl_tot)
    logger.info("-" * 70)
    logger.info("%-25s  %12s  %11.1f", "Avg iterations", "1.0", avg_iter)
    logger.info("%-25s  %12s  %11d", "Total replans", "0", total_replans)
    logger.info("=" * 70)

    q_delta = (sum(cl_q_scores) / max(len(cl_q_scores), 1)) - (sum(sp_q_scores) / max(len(sp_q_scores), 1))
    s_delta = (sum(cl_s_scores) / max(len(cl_s_scores), 1)) - (sum(sp_s_scores) / max(len(sp_s_scores), 1))
    logger.info(">> Quality improvement:  %+.3f avg score, %+d pass rate",
                q_delta, cl_q_pass - sp_q_pass)
    logger.info(">> Safety improvement:   %+.3f avg score, %+d pass rate",
                s_delta, cl_s_pass - sp_s_pass)

    # ---- 5. Per-case iteration detail ----
    logger.info("")
    logger.info("Per-case iteration detail (q=quality, s=safety):")
    for idx, cl_r in enumerate(cl_results):
        sigma = sigmas[idx]
        parts = []
        for it in cl_r.iterations:
            if not it.plan_tools:
                continue
            q = it.quality_verdict.get("score", 0)
            s = it.safety_verdict.get("score", 0)
            flag = "✓" if it.aggregate_passed else "✗"
            tools = "+".join(it.plan_tools)
            parts.append(f"iter#{it.iteration}[{tools}](q={q:.3f},s={s:.3f}){flag}")
        logger.info("  %s (σ=%.3f): %s",
                     cl_r.case_id, sigma, " → ".join(parts))
        if cl_r.total_iterations > 1:
            logger.info("    exit: %s | best=iter#%d", cl_r.exit_reason, cl_r.best_iteration)

    # ---- 6. Save report ----
    if args.output:
        from pathlib import Path
        report = {
            "config": {
                "num_cases": num_cases, "max_iterations": max_iter,
                "sigmas": [round(s, 4) for s in sigmas], "seed_start": seed_start,
            },
            "summary": {
                "single_pass": {
                    "diagnosis_accuracy": sp_ok / max(sp_tot, 1),
                    "quality_avg": sum(sp_q_scores) / max(len(sp_q_scores), 1),
                    "quality_pass_rate": sp_q_pass / max(sp_tot, 1),
                    "safety_avg": sum(sp_s_scores) / max(len(sp_s_scores), 1),
                    "safety_pass_rate": sp_s_pass / max(sp_tot, 1),
                },
                "closed_loop": {
                    "diagnosis_accuracy": cl_ok / max(cl_tot, 1),
                    "quality_avg": sum(cl_q_scores) / max(len(cl_q_scores), 1),
                    "quality_pass_rate": cl_q_pass / max(cl_tot, 1),
                    "safety_avg": sum(cl_s_scores) / max(len(cl_s_scores), 1),
                    "safety_pass_rate": cl_s_pass / max(cl_tot, 1),
                    "avg_iterations": avg_iter,
                    "total_replans": total_replans,
                },
            },
            "cases": {
                "single_pass": [r.to_dict() for r in sp_results],
                "closed_loop": [r.to_dict() for r in cl_results],
            },
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
