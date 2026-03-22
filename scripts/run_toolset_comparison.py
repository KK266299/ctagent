#!/usr/bin/env python3
# ============================================================================
# 模块职责: 工具集扩充对比实验 — 4 路交叉比较
#   1. old-sp:  single-pass + 旧工具 (denoise_gaussian)
#   2. old-cl:  closed-loop + 旧工具 (gaussian + sharpen replan)
#   3. exp-sp:  single-pass + 扩充工具 (TV, severity-adaptive weight)
#   4. exp-cl:  closed-loop + 扩充工具 (TV primary + ScoreAwareReplanner)
#
# Usage:
#   PYTHONPATH=. python scripts/run_toolset_comparison.py
#   PYTHONPATH=. python scripts/run_toolset_comparison.py --num-cases 10 --max-iter 4
# 参考: 4KAgent — ablation: tool pool comparison
# ============================================================================
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("toolset_comparison")
logger.setLevel(logging.INFO)


@dataclass
class ConfigResult:
    name: str
    diag_correct: int = 0
    diag_total: int = 0
    q_scores: list[float] = field(default_factory=list)
    s_scores: list[float] = field(default_factory=list)
    q_pass: int = 0
    s_pass: int = 0
    avg_iter: float = 1.0
    total_replans: int = 0

    @property
    def accuracy(self) -> float:
        return self.diag_correct / max(self.diag_total, 1)

    @property
    def q_avg(self) -> float:
        return sum(self.q_scores) / max(len(self.q_scores), 1)

    @property
    def s_avg(self) -> float:
        return sum(self.s_scores) / max(len(self.s_scores), 1)

    @property
    def agg_avg(self) -> float:
        return sum(min(q, s) for q, s in zip(self.q_scores, self.s_scores)) / max(len(self.q_scores), 1)


def build_pipelines(max_iter: int) -> dict[str, Any]:
    from src.degradations.detector import DegradationDetector
    from src.degradations.types import DegradationType, Severity
    from src.planner.rule_planner import RuleBasedPlanner
    from pipeline.single_pass import SinglePassPipeline
    from pipeline.agent_loop import ClosedLoopPipeline
    from pipeline.replan import (
        RuleBasedReplanner, ScoreAwareReplanner,
        OLD_STRATEGIES,
    )

    detector = DegradationDetector(config={
        "noise_thresholds": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    })

    old_rules = {
        (DegradationType.NOISE, Severity.MILD): ["denoise_gaussian"],
        (DegradationType.NOISE, Severity.MODERATE): ["denoise_gaussian"],
        (DegradationType.NOISE, Severity.SEVERE): ["denoise_gaussian"],
    }
    expanded_rules = {
        (DegradationType.NOISE, Severity.MILD): [
            ("denoise_tv", {"weight": 0.03}),
        ],
        (DegradationType.NOISE, Severity.MODERATE): [
            ("denoise_tv", {"weight": 0.06}),
        ],
        (DegradationType.NOISE, Severity.SEVERE): [
            ("denoise_tv", {"weight": 0.10}),
        ],
    }

    old_planner = RuleBasedPlanner(rules=old_rules)
    exp_planner = RuleBasedPlanner(rules=expanded_rules)

    old_replanner = RuleBasedReplanner(strategies=OLD_STRATEGIES)
    smart_replanner = ScoreAwareReplanner()

    return {
        "old-sp": SinglePassPipeline(detector=detector, planner=old_planner),
        "old-cl": ClosedLoopPipeline(
            detector=detector, planner=old_planner,
            replanner=old_replanner, max_iterations=max_iter,
        ),
        "exp-sp": SinglePassPipeline(detector=detector, planner=exp_planner),
        "exp-cl": ClosedLoopPipeline(
            detector=detector, planner=exp_planner,
            replanner=smart_replanner, max_iterations=max_iter,
        ),
    }


def collect_sp_stats(cr: ConfigResult, r: Any) -> None:
    cr.diag_total += 1
    if r.diagnosis_correct:
        cr.diag_correct += 1
    cr.q_scores.append(r.quality_verdict.get("score", 0))
    cr.s_scores.append(r.safety_verdict.get("score", 0))
    if r.quality_verdict.get("passed"):
        cr.q_pass += 1
    if r.safety_verdict.get("passed"):
        cr.s_pass += 1


def collect_cl_stats(cr: ConfigResult, r: Any) -> None:
    cr.diag_total += 1
    if r.diagnosis_correct:
        cr.diag_correct += 1
    cr.q_scores.append(r.final_quality.get("score", 0))
    cr.s_scores.append(r.final_safety.get("score", 0))
    if r.final_quality.get("passed"):
        cr.q_pass += 1
    if r.final_safety.get("passed"):
        cr.s_pass += 1
    cr.total_replans += r.replan_count


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-Agent: Old vs Expanded Toolset Comparison")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    num_cases = args.num_cases
    seed_start = args.seed

    sigma_min, sigma_max = 0.03, 0.15
    sigmas = [sigma_min + (sigma_max - sigma_min) * i / max(num_cases - 1, 1)
              for i in range(num_cases)]

    logger.info("=" * 80)
    logger.info("CT-Agent: Old vs Expanded Toolset Comparison")
    logger.info("  cases=%d  sigma=[%.3f-%.3f]  max_iter=%d  seed=%d",
                num_cases, sigmas[0], sigmas[-1], args.max_iter, seed_start)
    logger.info("  Expanded toolset: TV (severity-adaptive) + ScoreAwareReplanner")
    logger.info("=" * 80)

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

    pipelines = build_pipelines(args.max_iter)

    stats = {k: ConfigResult(name=k) for k in ["old-sp", "old-cl", "exp-sp", "exp-cl"]}
    cl_iterations: dict[str, list[float]] = {"old-cl": [], "exp-cl": []}
    per_case: list[dict[str, Any]] = []
    cl_details: list[Any] = []

    for idx, case in enumerate(cases):
        cid = case["case_id"]
        sigma = sigmas[idx]
        case_row: dict[str, Any] = {"case_id": cid, "sigma": round(sigma, 4)}

        r_old_sp = pipelines["old-sp"].run_toy_case(case)
        collect_sp_stats(stats["old-sp"], r_old_sp)

        r_old_cl = pipelines["old-cl"].run_toy_case(case)
        collect_cl_stats(stats["old-cl"], r_old_cl)
        cl_iterations["old-cl"].append(r_old_cl.total_iterations)

        r_exp_sp = pipelines["exp-sp"].run_toy_case(case)
        collect_sp_stats(stats["exp-sp"], r_exp_sp)

        r_exp_cl = pipelines["exp-cl"].run_toy_case(case)
        collect_cl_stats(stats["exp-cl"], r_exp_cl)
        cl_iterations["exp-cl"].append(r_exp_cl.total_iterations)
        cl_details.append(r_exp_cl)

        case_row["old_sp_q"] = round(r_old_sp.quality_verdict.get("score", 0), 3)
        case_row["old_sp_s"] = round(r_old_sp.safety_verdict.get("score", 0), 3)
        case_row["exp_sp_q"] = round(r_exp_sp.quality_verdict.get("score", 0), 3)
        case_row["exp_sp_s"] = round(r_exp_sp.safety_verdict.get("score", 0), 3)
        case_row["exp_cl_q"] = round(r_exp_cl.final_quality.get("score", 0), 3)
        case_row["exp_cl_s"] = round(r_exp_cl.final_safety.get("score", 0), 3)
        case_row["exp_cl_iter"] = r_exp_cl.total_iterations
        case_row["old_sp_diag"] = r_old_sp.diagnosis_correct
        case_row["exp_sp_diag"] = r_exp_sp.diagnosis_correct
        case_row["exp_cl_diag"] = r_exp_cl.diagnosis_correct
        per_case.append(case_row)

    stats["old-cl"].avg_iter = sum(cl_iterations["old-cl"]) / max(len(cl_iterations["old-cl"]), 1)
    stats["exp-cl"].avg_iter = sum(cl_iterations["exp-cl"]) / max(len(cl_iterations["exp-cl"]), 1)

    # ---- Summary Table ----
    keys = ["old-sp", "old-cl", "exp-sp", "exp-cl"]
    logger.info("")
    logger.info("=" * 80)
    logger.info("TOOLSET COMPARISON: 4-Way Results")
    logger.info("-" * 80)
    logger.info("%-25s  %10s  %10s  %10s  %10s", "", *keys)
    logger.info("-" * 80)

    def row(label: str, vals: list[str]) -> None:
        logger.info("%-25s  %10s  %10s  %10s  %10s", label, *vals)

    row("Diagnosis correct", [f"{stats[k].diag_correct}/{stats[k].diag_total}" for k in keys])
    row("Diagnosis accuracy", [f"{stats[k].accuracy:.0%}" for k in keys])
    logger.info("-" * 80)
    row("Quality score (avg)", [f"{stats[k].q_avg:.3f}" for k in keys])
    row("Quality pass", [f"{stats[k].q_pass}/{stats[k].diag_total}" for k in keys])
    row("Safety score (avg)", [f"{stats[k].s_avg:.3f}" for k in keys])
    row("Safety pass", [f"{stats[k].s_pass}/{stats[k].diag_total}" for k in keys])
    row("Aggregate (avg)", [f"{stats[k].agg_avg:.3f}" for k in keys])
    logger.info("-" * 80)
    row("Avg iterations", [
        "1.0", f"{stats['old-cl'].avg_iter:.1f}",
        "1.0", f"{stats['exp-cl'].avg_iter:.1f}",
    ])
    row("Total replans", [
        "0", str(stats["old-cl"].total_replans),
        "0", str(stats["exp-cl"].total_replans),
    ])
    logger.info("=" * 80)

    # ---- Delta Analysis ----
    logger.info("")
    logger.info("DELTA ANALYSIS:")
    logger.info("  1) Tool upgrade only (old-sp → exp-sp):")
    logger.info("     Quality:    %+.3f avg,  %+d pass",
                stats["exp-sp"].q_avg - stats["old-sp"].q_avg,
                stats["exp-sp"].q_pass - stats["old-sp"].q_pass)
    logger.info("     Safety:     %+.3f avg,  %+d pass",
                stats["exp-sp"].s_avg - stats["old-sp"].s_avg,
                stats["exp-sp"].s_pass - stats["old-sp"].s_pass)
    logger.info("     Aggregate:  %+.3f avg",
                stats["exp-sp"].agg_avg - stats["old-sp"].agg_avg)
    logger.info("     Diagnosis:  %+.0f%% (%d → %d)",
                100 * (stats["exp-sp"].accuracy - stats["old-sp"].accuracy),
                stats["old-sp"].diag_correct, stats["exp-sp"].diag_correct)

    logger.info("")
    logger.info("  2) Closed-loop benefit (exp-sp → exp-cl):")
    logger.info("     Quality:    %+.3f avg,  %+d pass",
                stats["exp-cl"].q_avg - stats["exp-sp"].q_avg,
                stats["exp-cl"].q_pass - stats["exp-sp"].q_pass)
    logger.info("     Safety:     %+.3f avg,  %+d pass",
                stats["exp-cl"].s_avg - stats["exp-sp"].s_avg,
                stats["exp-cl"].s_pass - stats["exp-sp"].s_pass)
    logger.info("     Aggregate:  %+.3f avg",
                stats["exp-cl"].agg_avg - stats["exp-sp"].agg_avg)
    logger.info("     Diagnosis:  %+.0f%% (%d → %d)",
                100 * (stats["exp-cl"].accuracy - stats["exp-sp"].accuracy),
                stats["exp-sp"].diag_correct, stats["exp-cl"].diag_correct)

    logger.info("")
    logger.info("  3) Full upgrade (old-sp → exp-cl):")
    logger.info("     Quality:    %+.3f avg,  %+d pass",
                stats["exp-cl"].q_avg - stats["old-sp"].q_avg,
                stats["exp-cl"].q_pass - stats["old-sp"].q_pass)
    logger.info("     Safety:     %+.3f avg,  %+d pass",
                stats["exp-cl"].s_avg - stats["old-sp"].s_avg,
                stats["exp-cl"].s_pass - stats["old-sp"].s_pass)
    logger.info("     Aggregate:  %+.3f avg",
                stats["exp-cl"].agg_avg - stats["old-sp"].agg_avg)
    logger.info("     Diagnosis:  %+.0f%% (%d → %d)",
                100 * (stats["exp-cl"].accuracy - stats["old-sp"].accuracy),
                stats["old-sp"].diag_correct, stats["exp-cl"].diag_correct)

    # ---- Per-case Detail ----
    logger.info("")
    logger.info("PER-CASE DETAIL:")
    logger.info("  %-18s  %8s  %8s  %8s  %8s  %8s  %8s  %4s  %12s",
                "case", "old_q", "old_s", "exp_q", "exp_s", "cl_q", "cl_s", "iter", "diag o→e→c")
    for pc in per_case:
        logger.info("  %-18s  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %4d  %s→%s→%s",
                     f"{pc['case_id']}(σ={pc['sigma']:.3f})",
                     pc["old_sp_q"], pc["old_sp_s"],
                     pc["exp_sp_q"], pc["exp_sp_s"],
                     pc["exp_cl_q"], pc["exp_cl_s"],
                     pc["exp_cl_iter"],
                     "✓" if pc["old_sp_diag"] else "✗",
                     "✓" if pc["exp_sp_diag"] else "✗",
                     "✓" if pc["exp_cl_diag"] else "✗")

    # ---- Closed-loop iteration trace ----
    logger.info("")
    logger.info("EXP-CL ITERATION TRACE:")
    for idx, cl_r in enumerate(cl_details):
        sigma = sigmas[idx]
        parts = []
        for it in cl_r.iterations:
            if not it.plan_tools:
                if it.replan_decision:
                    parts.append(f"→{it.replan_decision}({it.replan_reason})")
                continue
            q = it.quality_verdict.get("score", 0)
            s = it.safety_verdict.get("score", 0)
            flag = "✓" if it.aggregate_passed else "✗"
            tools = "+".join(it.plan_tools)
            parts.append(f"iter#{it.iteration}[{tools}](q={q:.3f},s={s:.3f}){flag}")
        trace_str = " → ".join(parts)
        logger.info("  %s (σ=%.3f): %s", cl_r.case_id, sigma, trace_str)
        if cl_r.total_iterations > 1:
            logger.info("    exit: %s | best=iter#%d (score=%.3f)",
                        cl_r.exit_reason, cl_r.best_iteration, cl_r.best_score)

    # ---- Save Report ----
    if args.output:
        from pathlib import Path
        report = {
            "config": {
                "num_cases": num_cases, "max_iterations": args.max_iter,
                "sigmas": [round(s, 4) for s in sigmas], "seed_start": seed_start,
            },
            "summary": {k: {
                "accuracy": round(s.accuracy, 4),
                "q_avg": round(s.q_avg, 4), "q_pass": s.q_pass,
                "s_avg": round(s.s_avg, 4), "s_pass": s.s_pass,
                "agg_avg": round(s.agg_avg, 4),
                "avg_iter": round(s.avg_iter, 2), "replans": s.total_replans,
            } for k, s in stats.items()},
            "per_case": per_case,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
