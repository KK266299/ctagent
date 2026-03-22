#!/usr/bin/env python3
# ============================================================================
# 模块职责: 正式比较实验 — 5 路交叉对比 + text vs vision 消融
#   1. rule-sp:    SinglePass + RuleBasedPlanner (TV-adaptive)
#   2. rule-cl:    ClosedLoop + ScoreAwareReplanner
#   3. api-mock:   ClosedLoop + APIGuidedPlanner (mock client)
#   4. api-text:   ClosedLoop + APIGuidedPlanner (real, text-only)
#   5. api-vision: ClosedLoop + APIGuidedPlanner (real, vision-first)
#
# 输出:
#   summary.json / summary.csv / trajectories.json / case_studies.json
#
# Usage:
#   PYTHONPATH=. python scripts/run_formal_comparison.py
#   PYTHONPATH=. python scripts/run_formal_comparison.py --skip-real-api
#   OPENAI_API_KEY=sk-... PYTHONPATH=. python scripts/run_formal_comparison.py
# ============================================================================
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("formal_comparison")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Mock LLM Client — 确定性响应, 不调用真实 API
# ---------------------------------------------------------------------------

class ScriptedMockClient:
    """确定性 mock client, 根据 iteration 返回固定 JSON。"""

    RESPONSES = [
        '{"decision":"retry","steps":[{"tool_name":"denoise_tv","params":{"weight":0.08}}],"reason":"mock: moderate TV for initial denoise"}',
        '{"decision":"retry","steps":[{"tool_name":"denoise_bilateral","params":{"sigma_color":0.04,"sigma_spatial":4}}],"reason":"mock: try bilateral"}',
        '{"decision":"retry","steps":[{"tool_name":"denoise_nlm"}],"reason":"mock: try NLM"}',
        '{"decision":"stop","reason":"mock: exhausted strategies"}',
    ]

    def __init__(self) -> None:
        self._call_idx = 0

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        resp_text = self.RESPONSES[min(self._call_idx, len(self.RESPONSES) - 1)]
        self._call_idx += 1

        class _R:
            text = resp_text
            model = "mock"
            usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
            raw = None
            finish_reason = "stop"
        return _R()

    def chat_with_image(self, text: str, image_b64: str, system_prompt: str = "", **kwargs: Any) -> Any:
        return self.chat([{"role": "user", "content": text}], **kwargs)

    def reset(self) -> None:
        self._call_idx = 0


# ---------------------------------------------------------------------------
# 指标收集
# ---------------------------------------------------------------------------

@dataclass
class ConfigStats:
    name: str
    diag_correct: int = 0
    diag_total: int = 0
    q_scores: list[float] = field(default_factory=list)
    s_scores: list[float] = field(default_factory=list)
    q_pass: int = 0
    s_pass: int = 0
    total_replans: int = 0
    total_iterations_sum: float = 0.0
    all_tools: list[str] = field(default_factory=list)
    fallback_count: int = 0
    skipped: bool = False

    @property
    def n(self) -> int:
        return max(self.diag_total, 1)

    @property
    def accuracy(self) -> float:
        return self.diag_correct / self.n

    @property
    def q_avg(self) -> float:
        return sum(self.q_scores) / max(len(self.q_scores), 1)

    @property
    def s_avg(self) -> float:
        return sum(self.s_scores) / max(len(self.s_scores), 1)

    @property
    def agg_avg(self) -> float:
        return sum(min(q, s) for q, s in zip(self.q_scores, self.s_scores)) / max(len(self.q_scores), 1)

    @property
    def avg_iter(self) -> float:
        return self.total_iterations_sum / self.n

    @property
    def tool_diversity(self) -> int:
        return len(set(self.all_tools))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "diagnosis_accuracy": round(self.accuracy, 4),
            "diagnosis_correct": self.diag_correct,
            "diagnosis_total": self.diag_total,
            "quality_avg": round(self.q_avg, 4),
            "quality_pass": self.q_pass,
            "safety_avg": round(self.s_avg, 4),
            "safety_pass": self.s_pass,
            "aggregate_avg": round(self.agg_avg, 4),
            "avg_iterations": round(self.avg_iter, 2),
            "replan_count": self.total_replans,
            "tool_diversity": self.tool_diversity,
            "unique_tools": sorted(set(self.all_tools)),
            "fallback_count": self.fallback_count,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Pipeline 构建
# ---------------------------------------------------------------------------

def build_pipelines(
    max_iter: int,
    skip_real_api: bool,
    llm_kwargs: dict[str, Any],
    vision_llm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.degradations.detector import DegradationDetector
    from src.degradations.types import DegradationType, Severity
    from src.planner.rule_planner import RuleBasedPlanner
    from pipeline.single_pass import SinglePassPipeline
    from pipeline.agent_loop import ClosedLoopPipeline
    from pipeline.replan import ScoreAwareReplanner
    from pipeline.api_guided_planner import APIGuidedPlanner

    detector = DegradationDetector(config={
        "noise_thresholds": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    })

    expanded_rules = {
        (DegradationType.NOISE, Severity.MILD): [("denoise_tv", {"weight": 0.03})],
        (DegradationType.NOISE, Severity.MODERATE): [("denoise_tv", {"weight": 0.06})],
        (DegradationType.NOISE, Severity.SEVERE): [("denoise_tv", {"weight": 0.10})],
    }
    rule_planner = RuleBasedPlanner(rules=expanded_rules)
    score_replanner = ScoreAwareReplanner()

    pipelines: dict[str, Any] = {}

    # 1. rule-sp
    pipelines["rule-sp"] = SinglePassPipeline(detector=detector, planner=rule_planner)

    # 2. rule-cl
    pipelines["rule-cl"] = ClosedLoopPipeline(
        detector=detector, planner=rule_planner,
        replanner=score_replanner, max_iterations=max_iter,
    )

    # 3. api-mock
    mock_client = ScriptedMockClient()
    mock_planner = APIGuidedPlanner(
        llm_client=mock_client, mode="full",
        fallback_planner=rule_planner, fallback_replanner=score_replanner,
    )
    pipelines["api-mock"] = ClosedLoopPipeline(
        detector=detector, planner=mock_planner, replanner=mock_planner,
        max_iterations=max_iter, pass_image_to_planner=True,
    )

    # 4 & 5. api-text / api-vision (需要真实 API)
    if not skip_real_api:
        from llm.api_client import LLMConfig, create_client
        config = LLMConfig(**llm_kwargs)
        real_client = create_client(config)

        text_planner = APIGuidedPlanner(
            llm_client=real_client, mode="full", use_vision=False,
            fallback_planner=rule_planner, fallback_replanner=score_replanner,
        )
        pipelines["api-text"] = ClosedLoopPipeline(
            detector=detector, planner=text_planner, replanner=text_planner,
            max_iterations=max_iter, pass_image_to_planner=False,
        )

        v_kwargs = vision_llm_kwargs or llm_kwargs
        v_config = LLMConfig(**v_kwargs)
        vision_client = create_client(v_config) if v_kwargs is not llm_kwargs else real_client

        vision_planner = APIGuidedPlanner(
            llm_client=vision_client, mode="full", use_vision=True,
            fallback_planner=rule_planner, fallback_replanner=score_replanner,
        )
        pipelines["api-vision"] = ClosedLoopPipeline(
            detector=detector, planner=vision_planner, replanner=vision_planner,
            max_iterations=max_iter, pass_image_to_planner=True,
        )

    return pipelines


# ---------------------------------------------------------------------------
# 单 case 运行 + 收集
# ---------------------------------------------------------------------------

def run_single_case(pipeline: Any, case: dict[str, Any], config_name: str) -> dict[str, Any]:
    """运行单 case, 返回统一格式的 per-case 记录。"""
    planner = getattr(pipeline, "planner", None)
    if hasattr(planner, "reset_records"):
        planner.reset_records()

    if hasattr(pipeline, "run_toy_case"):
        r = pipeline.run_toy_case(case)
    else:
        raise RuntimeError(f"Pipeline {config_name} has no run_toy_case")

    is_cl = hasattr(r, "total_iterations")

    record: dict[str, Any] = {
        "config": config_name,
        "case_id": r.case_id,
        "diagnosis_correct": r.diagnosis_correct,
        "diagnosis": r.diagnosis,
    }

    if is_cl:
        record["total_iterations"] = r.total_iterations
        record["replan_count"] = r.replan_count
        record["best_iteration"] = r.best_iteration
        record["best_score"] = round(r.best_score, 4)
        record["exit_reason"] = r.exit_reason
        record["quality_score"] = r.final_quality.get("score", 0)
        record["quality_passed"] = r.final_quality.get("passed", False)
        record["safety_score"] = r.final_safety.get("score", 0)
        record["safety_passed"] = r.final_safety.get("passed", False)
        all_tools = []
        iterations_detail = []
        for it in r.iterations:
            all_tools.extend(it.plan_tools)
            iterations_detail.append(it.to_dict())
        record["tools_used"] = all_tools
        record["iterations"] = iterations_detail
    else:
        record["total_iterations"] = 1
        record["replan_count"] = 0
        record["best_iteration"] = 0
        record["quality_score"] = r.quality_verdict.get("score", 0)
        record["quality_passed"] = r.quality_verdict.get("passed", False)
        record["safety_score"] = r.safety_verdict.get("score", 0)
        record["safety_passed"] = r.safety_verdict.get("passed", False)
        record["tools_used"] = r.plan_tools
        record["best_score"] = round(min(record["quality_score"], record["safety_score"]), 4)
        record["exit_reason"] = "single_pass"

    # API 专有记录
    if hasattr(planner, "call_records"):
        record["api_records"] = [cr.to_dict() for cr in planner.call_records]
        record["fallback_events"] = list(planner.fallback_events)
    else:
        record["api_records"] = []
        record["fallback_events"] = []

    return record


def update_stats(stats: ConfigStats, rec: dict[str, Any]) -> None:
    stats.diag_total += 1
    if rec["diagnosis_correct"]:
        stats.diag_correct += 1
    stats.q_scores.append(rec["quality_score"])
    stats.s_scores.append(rec["safety_score"])
    if rec["quality_passed"]:
        stats.q_pass += 1
    if rec["safety_passed"]:
        stats.s_pass += 1
    stats.total_replans += rec["replan_count"]
    stats.total_iterations_sum += rec["total_iterations"]
    stats.all_tools.extend(rec["tools_used"])
    stats.fallback_count += len(rec["fallback_events"])


# ---------------------------------------------------------------------------
# Case Studies 自动挑选
# ---------------------------------------------------------------------------

def select_case_studies(
    all_records: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    studies: dict[str, list[dict[str, Any]]] = {
        "rule_fail_api_success": [],
        "api_first_iter_pass": [],
        "api_replan_success": [],
        "api_fallback_case": [],
    }

    api_key = "api-vision" if "api-vision" in all_records else "api-mock"
    rule_key = "rule-cl"
    if rule_key not in all_records or api_key not in all_records:
        return studies

    rule_by_case = {r["case_id"]: r for r in all_records[rule_key]}
    api_by_case = {r["case_id"]: r for r in all_records[api_key]}

    for cid, api_rec in api_by_case.items():
        rule_rec = rule_by_case.get(cid)
        if not rule_rec:
            continue

        # rule fail + api success
        if not rule_rec["diagnosis_correct"] and api_rec["diagnosis_correct"]:
            studies["rule_fail_api_success"].append({
                "case_id": cid,
                "rule_cl": rule_rec,
                "api": api_rec,
                "reason": "rule-cl misdiagnosed, API-guided correct",
            })

        # api first iter pass
        iters = api_rec.get("iterations", [])
        if iters and len(iters) >= 1 and iters[0].get("aggregate_passed", False):
            studies["api_first_iter_pass"].append({
                "case_id": cid,
                "api": api_rec,
                "reason": "API-guided passed judge at iteration #0",
            })

        # api replan success
        if api_rec["replan_count"] > 0 and api_rec["diagnosis_correct"]:
            if iters and not iters[0].get("aggregate_passed", False):
                studies["api_replan_success"].append({
                    "case_id": cid,
                    "api": api_rec,
                    "reason": f"Failed iter#0, succeeded after {api_rec['replan_count']} replans",
                })

        # fallback case
        if api_rec.get("fallback_events"):
            studies["api_fallback_case"].append({
                "case_id": cid,
                "api": api_rec,
                "reason": f"Triggered {len(api_rec['fallback_events'])} fallback(s)",
            })

    return studies


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_summary_json(
    out_dir: Path,
    experiment_config: dict[str, Any],
    stats: dict[str, ConfigStats],
    config_keys: list[str],
) -> None:
    delta: dict[str, Any] = {}
    pairs = [
        ("rule_sp_vs_rule_cl", "rule-sp", "rule-cl"),
        ("rule_cl_vs_api_vision", "rule-cl", "api-vision"),
        ("rule_cl_vs_api_mock", "rule-cl", "api-mock"),
        ("api_text_vs_api_vision", "api-text", "api-vision"),
    ]
    for label, a, b in pairs:
        if a in stats and b in stats and not stats[a].skipped and not stats[b].skipped:
            delta[label] = {
                "accuracy_delta": round(stats[b].accuracy - stats[a].accuracy, 4),
                "quality_delta": round(stats[b].q_avg - stats[a].q_avg, 4),
                "safety_delta": round(stats[b].s_avg - stats[a].s_avg, 4),
                "aggregate_delta": round(stats[b].agg_avg - stats[a].agg_avg, 4),
            }

    data = {
        "experiment_config": experiment_config,
        "configs": {k: stats[k].to_dict() for k in config_keys},
        "delta_analysis": delta,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def write_summary_csv(
    out_dir: Path,
    stats: dict[str, ConfigStats],
    config_keys: list[str],
) -> None:
    fields = [
        "config", "diagnosis_accuracy", "quality_avg", "quality_pass",
        "safety_avg", "safety_pass", "aggregate_avg", "avg_iterations",
        "replan_count", "tool_diversity", "fallback_count", "skipped",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in config_keys:
            s = stats[k]
            w.writerow({
                "config": k,
                "diagnosis_accuracy": round(s.accuracy, 4),
                "quality_avg": round(s.q_avg, 4),
                "quality_pass": s.q_pass,
                "safety_avg": round(s.s_avg, 4),
                "safety_pass": s.s_pass,
                "aggregate_avg": round(s.agg_avg, 4),
                "avg_iterations": round(s.avg_iter, 2),
                "replan_count": s.total_replans,
                "tool_diversity": s.tool_diversity,
                "fallback_count": s.fallback_count,
                "skipped": s.skipped,
            })


def print_summary_table(stats: dict[str, ConfigStats], keys: list[str]) -> None:
    active = [k for k in keys if not stats[k].skipped]
    hdr = "%-22s" + "  %12s" * len(active)
    logger.info("")
    logger.info("=" * (22 + 14 * len(active)))
    logger.info("FORMAL COMPARISON RESULTS")
    logger.info("-" * (22 + 14 * len(active)))
    logger.info(hdr, "", *active)
    logger.info("-" * (22 + 14 * len(active)))

    def row(label: str, vals: list[str]) -> None:
        fmt = "%-22s" + "  %12s" * len(vals)
        logger.info(fmt, label, *vals)

    row("Diagnosis accuracy", [f"{stats[k].accuracy:.0%}" for k in active])
    row("  correct/total", [f"{stats[k].diag_correct}/{stats[k].diag_total}" for k in active])
    row("Quality avg", [f"{stats[k].q_avg:.3f}" for k in active])
    row("Quality pass", [f"{stats[k].q_pass}/{stats[k].diag_total}" for k in active])
    row("Safety avg", [f"{stats[k].s_avg:.3f}" for k in active])
    row("Safety pass", [f"{stats[k].s_pass}/{stats[k].diag_total}" for k in active])
    row("Aggregate avg", [f"{stats[k].agg_avg:.3f}" for k in active])
    row("Avg iterations", [f"{stats[k].avg_iter:.1f}" for k in active])
    row("Replan count", [str(stats[k].total_replans) for k in active])
    row("Tool diversity", [str(stats[k].tool_diversity) for k in active])
    row("Fallback count", [str(stats[k].fallback_count) for k in active])
    logger.info("=" * (22 + 14 * len(active)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CT-Agent: Formal Comparison Experiment")
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/formal_comparison")
    parser.add_argument("--skip-real-api", action="store_true", help="Skip api-text and api-vision configs")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat-v3-0324")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--vision-model", type=str, default="qwen/qwen-2.5-vl-72b-instruct",
                        help="Vision-capable model for api-vision config")
    parser.add_argument("--config", type=str, default=None, help="YAML config file (overrides CLI args)")
    args = parser.parse_args()

    # 从 YAML 加载配置 (若提供)
    yaml_vision_llm: dict[str, Any] | None = None
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        toy_cfg = cfg.get("toy", {})
        args.num_cases = toy_cfg.get("num_cases", args.num_cases)
        args.image_size = toy_cfg.get("image_size", args.image_size)
        args.seed = toy_cfg.get("seed", args.seed)
        llm_cfg = cfg.get("llm", {})
        args.model = llm_cfg.get("model", args.model)
        args.base_url = llm_cfg.get("base_url", args.base_url)
        args.max_iter = cfg.get("pipeline", {}).get("max_iterations", args.max_iter)
        args.output_dir = cfg.get("output", {}).get("dir", args.output_dir)
        if "vision_llm" in cfg:
            v = cfg["vision_llm"]
            args.vision_model = v.get("model", args.vision_model)
            yaml_vision_llm = {
                "provider": v.get("provider", "openai"),
                "model": v.get("model", args.vision_model),
                "base_url": v.get("base_url", args.base_url),
                "temperature": v.get("temperature", 0.1),
                "max_tokens": v.get("max_tokens", 1024),
            }
            if v.get("api_key"):
                yaml_vision_llm["api_key"] = v["api_key"]
        if llm_cfg.get("api_key"):
            os.environ.setdefault("OPENAI_API_KEY", llm_cfg["api_key"])

    api_key = os.environ.get("OPENAI_API_KEY", "")
    skip_real = args.skip_real_api or not api_key
    if skip_real and not args.skip_real_api:
        logger.warning("OPENAI_API_KEY not set, skipping api-text and api-vision configs")

    num_cases = args.num_cases
    sigma_min, sigma_max = 0.03, 0.15
    sigmas = [sigma_min + (sigma_max - sigma_min) * i / max(num_cases - 1, 1)
              for i in range(num_cases)]

    logger.info("=" * 80)
    logger.info("CT-Agent: Formal Comparison Experiment")
    logger.info("  cases=%d  sigma=[%.3f-%.3f]  max_iter=%d  seed=%d",
                num_cases, sigmas[0], sigmas[-1], args.max_iter, args.seed)
    logger.info("  model=%s  vision_model=%s  skip_real_api=%s", args.model, args.vision_model, skip_real)
    logger.info("=" * 80)

    # ---- 1. Generate cases ----
    from dataset.toy import generate_toy_case
    cases = []
    for i in range(num_cases):
        case = generate_toy_case(
            size=args.image_size,
            degradation="noise",
            degradation_params={"sigma": sigmas[i]},
            seed=args.seed + i,
        )
        cases.append((case, sigmas[i]))

    # ---- 2. Build pipelines ----
    llm_kwargs = {
        "provider": "openai",
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    vision_llm_kwargs: dict[str, Any] | None = yaml_vision_llm if args.config else None
    if vision_llm_kwargs is None and args.vision_model and args.vision_model != args.model:
        vision_llm_kwargs = {
            "provider": "openai",
            "model": args.vision_model,
            "base_url": args.base_url,
            "temperature": 0.1,
            "max_tokens": 1024,
        }

    pipelines = build_pipelines(args.max_iter, skip_real, llm_kwargs, vision_llm_kwargs)

    config_keys = ["rule-sp", "rule-cl", "api-mock", "api-text", "api-vision"]
    stats = {k: ConfigStats(name=k) for k in config_keys}
    all_records: dict[str, list[dict[str, Any]]] = {k: [] for k in config_keys}

    for k in config_keys:
        if k not in pipelines:
            stats[k].skipped = True

    # ---- 3. Run experiment ----
    t0 = time.time()
    for idx, (case, sigma) in enumerate(cases):
        cid = case["case_id"]
        logger.info("[%d/%d] %s  sigma=%.3f", idx + 1, num_cases, cid, sigma)

        for cfg_name in config_keys:
            if cfg_name not in pipelines:
                continue

            pipe = pipelines[cfg_name]

            # mock client 需要在每个 case 前 reset
            if cfg_name == "api-mock":
                mock_pl = pipe.planner
                if hasattr(mock_pl, "caller") and mock_pl.caller:
                    client = mock_pl.caller.llm_client
                    if hasattr(client, "reset"):
                        client.reset()

            rec = run_single_case(pipe, case, cfg_name)
            rec["sigma"] = round(sigma, 4)
            all_records[cfg_name].append(rec)
            update_stats(stats[cfg_name], rec)

    elapsed = time.time() - t0
    logger.info("Total experiment time: %.1fs", elapsed)

    # ---- 4. Print summary ----
    print_summary_table(stats, config_keys)

    # ---- 5. Delta analysis (console) ----
    logger.info("")
    logger.info("ABLATION: api-text vs api-vision")
    if not stats["api-text"].skipped and not stats["api-vision"].skipped:
        logger.info("  Accuracy:   text=%.0f%%  vision=%.0f%%  delta=%+.0f%%",
                     100 * stats["api-text"].accuracy, 100 * stats["api-vision"].accuracy,
                     100 * (stats["api-vision"].accuracy - stats["api-text"].accuracy))
        logger.info("  Aggregate:  text=%.3f  vision=%.3f  delta=%+.3f",
                     stats["api-text"].agg_avg, stats["api-vision"].agg_avg,
                     stats["api-vision"].agg_avg - stats["api-text"].agg_avg)
        logger.info("  Tool div:   text=%d  vision=%d",
                     stats["api-text"].tool_diversity, stats["api-vision"].tool_diversity)
    else:
        logger.info("  (skipped — real API not available)")

    # ---- 6. Case studies ----
    case_studies = select_case_studies(all_records)
    for cat, items in case_studies.items():
        if items:
            logger.info("Case study [%s]: %d case(s) — e.g. %s", cat, len(items), items[0]["case_id"])

    # ---- 7. Write output files ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "num_cases": num_cases,
        "max_iterations": args.max_iter,
        "sigmas": [round(s, 4) for s in sigmas],
        "seed": args.seed,
        "model": args.model,
        "skip_real_api": skip_real,
        "elapsed_seconds": round(elapsed, 1),
    }

    write_summary_json(out_dir, experiment_config, stats, config_keys)
    write_summary_csv(out_dir, stats, config_keys)

    with open(out_dir / "trajectories.json", "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False, default=str)

    with open(out_dir / "case_studies.json", "w") as f:
        json.dump(case_studies, f, indent=2, ensure_ascii=False, default=str)

    logger.info("")
    logger.info("Output files:")
    for name in ["summary.json", "summary.csv", "trajectories.json", "case_studies.json"]:
        logger.info("  %s", out_dir / name)


if __name__ == "__main__":
    main()
