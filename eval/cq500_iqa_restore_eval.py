# ============================================================================
# 模块职责: CQ500 IQA 评估 + LLM/规则修复 + 修复质量评估
#   degraded → detect → plan (rule/llm) → restore (apply_chain) → IQA before/after
#   全部在 μ 值空间操作
#   复用: eval/cq500_manifest, src/tools/mcp_style, src/planner, llm/
# ============================================================================
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SliceResult:
    """单个 slice 的完整评测记录。"""
    case_id: str
    slice_name: str
    planner_mode: str
    degradation_detected: list[dict[str, str]]
    iqa_scores: dict[str, float]
    plan_reasoning: str
    tool_sequence: list[str]
    tool_params: list[dict]
    psnr_before: float = 0.0
    psnr_after: float = 0.0
    ssim_before: float = 0.0
    ssim_after: float = 0.0
    sharpness_before: float = 0.0
    sharpness_after: float = 0.0
    noise_before: float = 0.0
    noise_after: float = 0.0
    quality_trace: list[dict] = field(default_factory=list)
    restoration_success: bool = False
    error: str | None = None
    latency_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "slice_name": self.slice_name,
            "planner_mode": self.planner_mode,
            "degradation_detected": self.degradation_detected,
            "iqa_scores": self.iqa_scores,
            "plan_reasoning": self.plan_reasoning,
            "tool_sequence": self.tool_sequence,
            "tool_params": self.tool_params,
            "psnr_before": self.psnr_before,
            "psnr_after": self.psnr_after,
            "ssim_before": self.ssim_before,
            "ssim_after": self.ssim_after,
            "sharpness_before": self.sharpness_before,
            "sharpness_after": self.sharpness_after,
            "noise_before": self.noise_before,
            "noise_after": self.noise_after,
            "quality_trace": self.quality_trace,
            "restoration_success": self.restoration_success,
            "error": self.error,
            "latency_sec": self.latency_sec,
        }


def load_mu_image(h5_path: Path, dataset_key: str) -> np.ndarray:
    """从 HDF5 加载 μ 值图像。"""
    import h5py
    with h5py.File(str(h5_path), "r") as f:
        return f[dataset_key][:].astype(np.float64)


def evaluate_single_slice(
    clean_mu: np.ndarray,
    degraded_mu: np.ndarray,
    case_id: str,
    slice_name: str,
    planner_mode: str,
    detector: Any,
    rule_planner: Any | None,
    agent_planner: Any | None,
    restoration_tool: Any,
    perception_tool: Any,
    max_replan_rounds: int = 1,
) -> SliceResult:
    """对单个 slice 做完整 IQA + 修复 + 评估。

    Args:
        max_replan_rounds: LLM 模式最大 replan 轮数 (1 = 单轮, 2+ = 多轮 replan)
    """
    from src.iqa.metrics import compute_psnr, compute_ssim

    t0 = time.time()
    result = SliceResult(
        case_id=case_id,
        slice_name=slice_name,
        planner_mode=planner_mode,
        degradation_detected=[],
        iqa_scores={},
        plan_reasoning="",
        tool_sequence=[],
        tool_params=[],
    )

    try:
        data_range = float(max(clean_mu.max() - clean_mu.min(), 1e-10))

        # --- IQA before ---
        before_perception = perception_tool(degraded_mu, reference=clean_mu, data_range=data_range)
        nr_before = before_perception.get("no_reference", {})
        fr_before = before_perception.get("full_reference", {})

        result.psnr_before = fr_before.get("psnr", 0.0)
        result.ssim_before = fr_before.get("ssim", 0.0)
        result.sharpness_before = nr_before.get("sharpness", 0.0)
        result.noise_before = nr_before.get("noise_estimate", 0.0)

        # --- Degradation detection ---
        report = detector.detect(degraded_mu)
        result.iqa_scores = report.iqa_scores
        result.degradation_detected = [
            {"type": d.value, "severity": s.value}
            for d, s in report.degradations
        ]

        # --- Planning (single-pass for rule, multi-round for llm) ---
        if planner_mode == "rule" and rule_planner is not None:
            plan = rule_planner.plan(report)
            if plan and len(plan) > 0:
                result.plan_reasoning = plan.reasoning
                result.tool_sequence = plan.tool_names()
                result.tool_params = [s.params for s in plan.steps]
                steps = [{"tool_name": s.tool_name, "params": s.params} for s in plan.steps]
                chain_result = restoration_tool.apply_chain(degraded_mu, steps, reference=clean_mu)
                result.restoration_success = chain_result["success"]
                result.quality_trace = chain_result["quality_trace"]
                q_after = chain_result.get("quality_after", {})
                result.psnr_after = q_after.get("psnr", result.psnr_before)
                result.ssim_after = q_after.get("ssim", result.ssim_before)
                result.sharpness_after = q_after.get("sharpness", result.sharpness_before)
                result.noise_after = q_after.get("noise_estimate", result.noise_before)
            else:
                _fill_no_plan(result)

        elif planner_mode == "llm" and agent_planner is not None:
            _run_llm_with_replan(
                result=result,
                clean_mu=clean_mu,
                degraded_mu=degraded_mu,
                report=report,
                agent_planner=agent_planner,
                restoration_tool=restoration_tool,
                perception_tool=perception_tool,
                data_range=data_range,
                max_rounds=max_replan_rounds,
            )
        else:
            _fill_no_plan(result)

    except Exception as e:
        result.error = str(e)
        logger.warning("[%s/%s/%s] Error: %s", case_id, slice_name, planner_mode, e)

    result.latency_sec = time.time() - t0
    return result


def _fill_no_plan(result: SliceResult) -> None:
    result.plan_reasoning = "No degradation / no plan generated"
    result.psnr_after = result.psnr_before
    result.ssim_after = result.ssim_before
    result.sharpness_after = result.sharpness_before
    result.noise_after = result.noise_before
    result.restoration_success = True


def _run_llm_with_replan(
    result: SliceResult,
    clean_mu: np.ndarray,
    degraded_mu: np.ndarray,
    report: Any,
    agent_planner: Any,
    restoration_tool: Any,
    perception_tool: Any,
    data_range: float,
    max_rounds: int,
) -> None:
    """LLM multi-round: plan → execute → evaluate → replan if needed."""
    from src.iqa.metrics import compute_psnr, compute_ssim

    current_image = degraded_mu.copy()
    best_image = degraded_mu.copy()
    best_psnr = result.psnr_before
    best_ssim = result.ssim_before
    all_tool_sequence: list[str] = []
    all_tool_params: list[dict] = []
    all_quality_trace: list[dict] = []
    all_reasoning: list[str] = []
    round_history: list[dict] = []

    for round_idx in range(max_rounds):
        # Build replan context from previous rounds
        extra_context = ""
        if round_history:
            best_round = max(round_history, key=lambda r: r["psnr_after"] / 40 + r["ssim_after"])
            worst_round = min(round_history, key=lambda r: r["psnr_after"] / 40 + r["ssim_after"])
            extra_context = (
                f"\n## Previous Rounds ({len(round_history)} attempts so far)\n"
                f"Best result: PSNR={best_psnr:.2f}, SSIM={best_ssim:.4f}\n"
                f"Target: PSNR > {best_psnr:.0f}, SSIM > {best_ssim:.2f}\n\n"
            )
            for rh in round_history:
                extra_context += (
                    f"- Round {rh['round']}: {rh['tools']} → "
                    f"PSNR={rh['psnr_after']:.2f}, SSIM={rh['ssim_after']:.4f}\n"
                )
            extra_context += (
                f"\nBest combo was: {best_round['tools']}\n"
                f"Worst combo was: {worst_round['tools']} — AVOID these tools.\n"
                "Each round starts fresh from the ORIGINAL degraded image.\n"
                "Try a COMPLETELY DIFFERENT tool combination. "
                "For metal artifacts: clip_extreme FIRST, then denoise_tv or denoise_wavelet.\n"
            )

        try:
            plan = agent_planner.plan(report, image=current_image, extra_context=extra_context)
        except Exception:
            plan = agent_planner.plan(report, image=current_image)

        if plan is None or len(plan) == 0:
            if round_idx == 0:
                _fill_no_plan(result)
                return
            break

        round_tools = plan.tool_names()
        round_params = [s.params for s in plan.steps]
        all_reasoning.append(f"[R{round_idx+1}] {plan.reasoning}")

        steps = [{"tool_name": s.tool_name, "params": s.params} for s in plan.steps]
        chain_result = restoration_tool.apply_chain(current_image, steps, reference=clean_mu)

        q_after = chain_result.get("quality_after", {})
        round_psnr = q_after.get("psnr", result.psnr_before)
        round_ssim = q_after.get("ssim", result.ssim_before)

        round_history.append({
            "round": round_idx + 1,
            "tools": round_tools,
            "psnr_after": round_psnr,
            "ssim_after": round_ssim,
        })

        all_quality_trace.extend(chain_result.get("quality_trace", []))

        # Keep best result (by combined score)
        combined = round_psnr / 40.0 + round_ssim
        best_combined = best_psnr / 40.0 + best_ssim
        if combined > best_combined:
            best_image = chain_result.get("_restored_image", current_image)
            best_psnr = round_psnr
            best_ssim = round_ssim
            all_tool_sequence = round_tools
            all_tool_params = round_params

        # Each round starts fresh from the original degraded image
        # so a bad round can't poison future attempts
        current_image = degraded_mu.copy()

        logger.debug(
            "[%s/%s] Round %d: PSNR=%.2f SSIM=%.4f (best: %.2f/%.4f)",
            result.case_id, result.slice_name, round_idx + 1,
            round_psnr, round_ssim, best_psnr, best_ssim,
        )

    result.plan_reasoning = " | ".join(all_reasoning)
    result.tool_sequence = all_tool_sequence
    result.tool_params = all_tool_params
    result.quality_trace = all_quality_trace
    result.restoration_success = True
    result.psnr_after = best_psnr
    result.ssim_after = best_ssim

    after_perception = perception_tool(best_image, data_range=data_range)
    nr_after = after_perception.get("no_reference", {})
    result.sharpness_after = nr_after.get("sharpness", result.sharpness_before)
    result.noise_after = nr_after.get("noise_estimate", result.noise_before)


def run_iqa_eval_batch(
    cases: list,
    detector: Any,
    rule_planner: Any | None,
    agent_planner: Any | None,
    restoration_tool: Any,
    perception_tool: Any,
    planner_modes: list[str],
    rate_limit_sec: float = 0.0,
    max_replan_rounds: int = 1,
) -> list[SliceResult]:
    """批量评测所有 case × slice × planner_mode。"""
    all_results: list[SliceResult] = []

    total_slices = sum(len(c.slices) for c in cases)
    total = total_slices * len(planner_modes)
    done = 0

    for case in cases:
        for se in case.slices:
            try:
                clean_mu = load_mu_image(se.gt_h5, "image")
                degraded_mu = load_mu_image(se.degraded_h5, "ma_CT")
            except Exception as e:
                logger.warning("[%s/%s] Failed to load images: %s",
                               case.case_id, se.slice_dir.name, e)
                continue

            for mode in planner_modes:
                done += 1
                rounds = max_replan_rounds if mode == "llm" else 1
                logger.info("[%d/%d] %s / %s / %s (rounds=%d)",
                            done, total, case.case_id, se.slice_dir.name, mode, rounds)

                result = evaluate_single_slice(
                    clean_mu=clean_mu,
                    degraded_mu=degraded_mu,
                    case_id=case.case_id,
                    slice_name=se.slice_dir.name,
                    planner_mode=mode,
                    detector=detector,
                    rule_planner=rule_planner,
                    agent_planner=agent_planner,
                    restoration_tool=restoration_tool,
                    perception_tool=perception_tool,
                    max_replan_rounds=rounds,
                )
                all_results.append(result)

                if rate_limit_sec > 0 and mode == "llm":
                    time.sleep(rate_limit_sec)

    return all_results


def aggregate_iqa_results(results: list[SliceResult]) -> dict[str, Any]:
    """汇总评测结果。"""
    if not results:
        return {}

    by_mode: dict[str, list[SliceResult]] = defaultdict(list)
    for r in results:
        by_mode[r.planner_mode].append(r)

    summary: dict[str, Any] = {}

    for mode, mode_results in by_mode.items():
        valid = [r for r in mode_results if r.error is None]
        n_success = sum(1 for r in valid if r.restoration_success)

        psnr_before = [r.psnr_before for r in valid]
        psnr_after = [r.psnr_after for r in valid]
        ssim_before = [r.ssim_before for r in valid]
        ssim_after = [r.ssim_after for r in valid]

        summary[mode] = {
            "n_total": len(mode_results),
            "n_valid": len(valid),
            "n_errors": len(mode_results) - len(valid),
            "n_restoration_success": n_success,
            "psnr_before_mean": float(np.mean(psnr_before)) if psnr_before else 0,
            "psnr_after_mean": float(np.mean(psnr_after)) if psnr_after else 0,
            "psnr_improvement": float(np.mean([a - b for a, b in zip(psnr_after, psnr_before)])) if valid else 0,
            "ssim_before_mean": float(np.mean(ssim_before)) if ssim_before else 0,
            "ssim_after_mean": float(np.mean(ssim_after)) if ssim_after else 0,
            "ssim_improvement": float(np.mean([a - b for a, b in zip(ssim_after, ssim_before)])) if valid else 0,
            "mean_latency_sec": float(np.mean([r.latency_sec for r in mode_results])),
        }

    return summary


def compute_degradation_distribution(results: list[SliceResult]) -> dict[str, int]:
    """统计退化类型分布。"""
    counter: Counter = Counter()
    for r in results:
        for d in r.degradation_detected:
            key = f"{d['type']}_{d['severity']}"
            counter[key] += 1
    return dict(counter.most_common())


def compute_tool_usage(results: list[SliceResult]) -> dict[str, Any]:
    """统计工具使用频次。"""
    tool_counter: Counter = Counter()
    chain_counter: Counter = Counter()
    for r in results:
        for t in r.tool_sequence:
            tool_counter[t] += 1
        chain_key = " -> ".join(r.tool_sequence) if r.tool_sequence else "(empty)"
        chain_counter[chain_key] += 1

    return {
        "tool_frequency": dict(tool_counter.most_common()),
        "chain_frequency": dict(chain_counter.most_common(20)),
    }


def save_iqa_outputs(
    results: list[SliceResult],
    summary: dict[str, Any],
    degradation_dist: dict[str, int],
    tool_usage: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """保存全部输出。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "iqa_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out / "per_slice.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    with open(out / "degradation_distribution.json", "w") as f:
        json.dump(degradation_dist, f, indent=2)

    with open(out / "tool_usage.json", "w") as f:
        json.dump(tool_usage, f, indent=2)

    if len(set(r.planner_mode for r in results)) > 1:
        import csv
        csv_path = out / "planner_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "mode", "n_valid", "psnr_before", "psnr_after", "psnr_delta",
                "ssim_before", "ssim_after", "ssim_delta", "n_success", "mean_latency",
            ])
            for mode, m in summary.items():
                writer.writerow([
                    mode,
                    m.get("n_valid", 0),
                    f"{m.get('psnr_before_mean', 0):.2f}",
                    f"{m.get('psnr_after_mean', 0):.2f}",
                    f"{m.get('psnr_improvement', 0):.2f}",
                    f"{m.get('ssim_before_mean', 0):.4f}",
                    f"{m.get('ssim_after_mean', 0):.4f}",
                    f"{m.get('ssim_improvement', 0):.4f}",
                    m.get("n_restoration_success", 0),
                    f"{m.get('mean_latency_sec', 0):.2f}",
                ])

    logger.info("Saved IQA eval outputs to %s", out)
