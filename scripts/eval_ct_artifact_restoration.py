#!/usr/bin/env python
"""CT 伪影检测→修复→评估 端到端流程。

在真实 CQ500 CT 上:
  1. 用 ct_artifact_simulator 生成 9 类伪影 + 金属伪影 (共 10 类)
  2. 用 DegradationDetector 或 LLM-based 检测器检测伪影类型
  3. 用 Planner 规划修复工具链 (rule-based 或 LLM-guided)
  4. 用 RestorationTool 执行修复
  5. 计算 PSNR/SSIM 评判修复效果
  6. 生成 4-panel 可视化对比图 (GT | degraded | restored | diff)

用法 (rule-based):
    PYTHONPATH=. bash -c '
    eval "$(conda shell.bash hook)" && conda activate llamafactory && \
    CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
        --planner rule \
        --output-dir /home/liuxinyao/project/ctagent/try_output/restoration \
        --gpu 1 --num-slices 2'

用法 (LLM-guided):
    export OPENAI_API_KEY="sk-or-v1-xxx"
    PYTHONPATH=. bash -c '
    eval "$(conda shell.bash hook)" && conda activate llamafactory && \
    CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_ct_artifact_restoration.py \
        --planner llm \
        --llm-model qwen/qwen-2.5-vl-72b-instruct \
        --llm-base-url https://openrouter.ai/api/v1 \
        --output-dir /home/liuxinyao/project/ctagent/try_output/restoration_llm \
        --gpu 1 --num-slices 2'
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_artifact_restoration")


MU_WATER = 0.192
WINDOWS = {"brain": (40, 80), "subdural": (75, 215)}


def hu_window_to_mu(wl_hu: float, ww_hu: float) -> tuple[float, float]:
    hu_lo = wl_hu - ww_hu / 2.0
    hu_hi = wl_hu + ww_hu / 2.0
    return MU_WATER * (1.0 + hu_lo / 1000.0), MU_WATER * (1.0 + hu_hi / 1000.0)


def mu_to_display(img: np.ndarray, window: str = "brain") -> np.ndarray:
    wl, ww = WINDOWS.get(window, (40, 80))
    lo, hi = hu_window_to_mu(wl, ww)
    return np.clip((img - lo) / max(hi - lo, 1e-10), 0.0, 1.0)


def mu_to_hu(mu: np.ndarray) -> np.ndarray:
    return (mu / MU_WATER - 1.0) * 1000.0


def find_middle_slices(processed_dir: str, num_slices: int = 2) -> list[Path]:
    root = Path(processed_dir)
    cases = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    selected: list[Path] = []
    for case_dir in cases:
        if len(selected) >= num_slices:
            break
        for series_dir in sorted(case_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            slice_dirs = sorted([d for d in series_dir.iterdir() if d.is_dir()])
            if len(slice_dirs) < 5:
                continue
            mid = len(slice_dirs) // 2
            gt_path = slice_dirs[mid] / "gt.h5"
            if gt_path.exists():
                selected.append(gt_path)
                break
    return selected


def find_all_slices(processed_dir: str) -> list[Path]:
    """收集 processed 目录下所有有 gt.h5 的 slice。"""
    root = Path(processed_dir)
    selected: list[Path] = []
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        for series_dir in sorted(patient_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            for slice_dir in sorted(series_dir.iterdir()):
                if not slice_dir.is_dir():
                    continue
                gt_path = slice_dir / "gt.h5"
                if gt_path.exists():
                    selected.append(gt_path)
    return selected


def plot_restoration_panel(
    gt_mu: np.ndarray,
    degraded_mu: np.ndarray,
    restored_mu: np.ndarray,
    artifact_type: str,
    severity: str,
    detection_result: str,
    tools_used: str,
    psnr_before: float,
    ssim_before: float,
    psnr_after: float,
    ssim_after: float,
    case_label: str,
    window: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    gt_disp = mu_to_display(gt_mu, window)
    deg_disp = mu_to_display(degraded_mu, window)
    res_disp = mu_to_display(restored_mu, window)
    diff_before = np.abs(degraded_mu - gt_mu)
    diff_after = np.abs(restored_mu - gt_mu)

    axes[0].imshow(gt_disp, cmap="gray")
    axes[0].set_title("GT (clean)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(deg_disp, cmap="gray")
    axes[1].set_title(
        f"Degraded ({artifact_type}/{severity})\n"
        f"PSNR={psnr_before:.2f}  SSIM={ssim_before:.4f}",
        fontsize=9,
    )
    axes[1].axis("off")

    axes[2].imshow(res_disp, cmap="gray")
    axes[2].set_title(
        f"Restored: {tools_used}\n"
        f"PSNR={psnr_after:.2f}  SSIM={ssim_after:.4f}",
        fontsize=9,
    )
    axes[2].axis("off")

    diff_max = max(np.percentile(diff_before, 99), np.percentile(diff_after, 99), 0.01)
    axes[3].imshow(diff_after, cmap="hot", vmin=0, vmax=diff_max)
    axes[3].set_title("|Restored − GT|", fontsize=10)
    axes[3].axis("off")

    detected_str = f"Detected: {detection_result}"
    fig.suptitle(
        f"{case_label} — {artifact_type}/{severity}  [{window}]  |  {detected_str}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="try_output/restoration")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-slices", type=int, default=2,
                        help="Number of slices (1 per patient). Ignored if --all-slices.")
    parser.add_argument("--all-slices", action="store_true",
                        help="Evaluate ALL slices from ALL patients (22k+).")
    parser.add_argument("--processed-dir", default="/home/liuxinyao/data/cq500_processed")
    parser.add_argument("--mat-dir", default="data/mar_physics")
    parser.add_argument("--save-images", action="store_true", default=None,
                        help="Force save visualization PNGs (default: auto based on slice count).")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Skip visualization PNGs to save disk/time.")
    # --- Planner mode ---
    parser.add_argument("--planner", choices=["rule", "llm"], default="rule",
                        help="Planner mode: rule (RuleBasedPlanner) or llm (LLM-guided AgentBasedPlanner)")
    # --- Detector mode ---
    parser.add_argument("--detector", choices=["rule", "llm"], default="rule",
                        help="Detector mode: rule (threshold-based) or llm (VLM-based artifact detection)")
    parser.add_argument("--llm-model", default="qwen/qwen-2.5-vl-72b-instruct",
                        help="LLM model name (for --planner llm or --detector llm)")
    parser.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1",
                        help="LLM API base URL (for --planner llm or --detector llm)")
    parser.add_argument("--llm-temperature", type=float, default=0.1,
                        help="LLM temperature (for --planner llm or --detector llm)")
    parser.add_argument("--llm-max-tokens", type=int, default=1024,
                        help="LLM max tokens (for --planner llm or --detector llm)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # --- imports that trigger tool registration ---
    from dataset.mar.ct_artifact_simulator import create_artifact_simulator
    from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig
    from dataset.mar.physics_params import PhysicsConfig, PhysicsParams
    from src.degradations.detector import DegradationDetector
    from src.planner.rule_planner import RuleBasedPlanner
    from src.tools.mcp_style.restoration_tool import RestorationTool

    import src.tools.classical.denoise       # noqa: F401
    import src.tools.classical.sharpen       # noqa: F401
    import src.tools.classical.deblur        # noqa: F401
    import src.tools.classical.clip          # noqa: F401
    import src.tools.classical.ring_removal  # noqa: F401
    import src.tools.classical.motion_correction       # noqa: F401
    import src.tools.classical.beam_hardening_correction  # noqa: F401
    import src.tools.classical.scatter_correction      # noqa: F401
    import src.tools.classical.truncation_correction   # noqa: F401

    try:
        import src.tools.learned.dncnn_tool  # noqa: F401
    except Exception:
        logger.warning("DnCNN tool not available, will skip if referenced")

    from src.iqa.metrics import compute_psnr, compute_ssim

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing CT geometry ...")
    geo = CTGeometry(CTGeometryConfig(
        image_size=416, num_angles=640, num_detectors=641, impl="astra_cuda",
    ))
    phy_cfg = PhysicsConfig(mat_dir=args.mat_dir)
    phy = PhysicsParams(phy_cfg)
    phy.load()

    restoration = RestorationTool()
    use_llm_planner = args.planner == "llm"
    use_llm_detector = args.detector == "llm"

    # --- LLM client (shared by planner and detector if both use llm) ---
    llm_client = None
    if use_llm_planner or use_llm_detector:
        from llm.api_client import LLMConfig, create_client
        llm_cfg = LLMConfig(
            provider="openai",
            model=args.llm_model,
            base_url=args.llm_base_url,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
        llm_client = create_client(llm_cfg)

    # --- Detector ---
    if use_llm_detector:
        from src.degradations.llm_detector import LLMDegradationDetector
        detector = LLMDegradationDetector(llm_client)
        logger.info("Using LLM-based detector: model=%s", args.llm_model)
    else:
        detector = DegradationDetector()
        logger.info("Using rule-based detector")

    use_llm = use_llm_planner

    if use_llm:
        from llm.planner_caller import PlannerCaller
        from src.planner.agent_based import AgentBasedPlanner

        planner_caller = PlannerCaller(llm_client=llm_client, max_steps=4)
        planner = AgentBasedPlanner(planner_caller=planner_caller, max_chain=4)
        logger.info("Using LLM-guided planner: model=%s, base_url=%s",
                     args.llm_model, args.llm_base_url)
    else:
        planner = RuleBasedPlanner(max_chain=4)
        logger.info("Using rule-based planner")

    if args.all_slices:
        gt_paths = find_all_slices(args.processed_dir)
    else:
        gt_paths = find_middle_slices(args.processed_dir, args.num_slices)
    logger.info("Selected %d slices for evaluation", len(gt_paths))

    do_save_images = args.save_images if args.save_images is not None else (not args.no_save_images and len(gt_paths) <= 50)

    artifact_types = [
        "ring", "motion", "beam_hardening", "scatter", "truncation",
        "low_dose", "sparse_view", "limited_angle", "focal_spot_blur",
    ]
    severities = ["mild", "moderate", "severe"]

    all_results: list[dict[str, Any]] = []

    for si, gt_path in enumerate(gt_paths):
        rel = gt_path.parent.relative_to(args.processed_dir)
        case_label = str(rel).replace("/", "_")
        logger.info("[%d/%d] %s", si + 1, len(gt_paths), case_label)

        with h5py.File(str(gt_path), "r") as f:
            mu_image = f["image"][:].astype(np.float64)
        hu_image = mu_to_hu(mu_image)

        for art_type in artifact_types:
            for sev in severities:
                logger.info("  %s/%s ...", art_type, sev)

                sim = create_artifact_simulator(art_type, geo, phy, seed=42 + si)
                sim_result = sim.simulate(hu_image, severity=sev)
                clean_recon = sim_result.poly_ct
                degraded = sim_result.artifact_results[0]["ma_CT"]

                data_range = float(max(clean_recon.max() - clean_recon.min(), 1e-10))
                psnr_before = compute_psnr(degraded, clean_recon, data_range)
                ssim_before = compute_ssim(degraded, clean_recon, data_range)

                report = detector.detect(degraded)
                detected_types = [
                    f"{dt.value}({sv.value})" for dt, sv in report.degradations
                ]
                detection_str = ", ".join(detected_types) if detected_types else "none"

                if use_llm:
                    plan = planner.plan(report, image=degraded)
                    planner_reasoning = plan.reasoning
                else:
                    plan = planner.plan(report)
                    planner_reasoning = plan.reasoning
                tools_str = "→".join(plan.tool_names()) if plan.tool_names() else "none"

                skipped = len(plan.steps) == 0
                if plan.steps:
                    steps = [{"tool_name": s.tool_name, "params": s.params} for s in plan.steps]
                    chain_result = restoration.apply_chain(degraded, steps, reference=clean_recon)
                    restored = chain_result["_restored_image"]
                else:
                    restored = degraded.copy()

                psnr_after = compute_psnr(restored, clean_recon, data_range)
                ssim_after = compute_ssim(restored, clean_recon, data_range)

                record: dict[str, Any] = {
                    "case": case_label,
                    "artifact_type": art_type,
                    "severity": sev,
                    "planner": args.planner,
                    "detected": detection_str,
                    "tools": tools_str,
                    "reasoning": planner_reasoning,
                    "skipped": skipped,
                    "psnr_before": round(psnr_before, 2),
                    "ssim_before": round(ssim_before, 4),
                    "psnr_after": round(psnr_after, 2),
                    "ssim_after": round(ssim_after, 4),
                    "psnr_delta": round(psnr_after - psnr_before, 2),
                    "ssim_delta": round(ssim_after - ssim_before, 4),
                }
                all_results.append(record)

                logger.info(
                    "    Detected: %s | Tools: %s | PSNR: %.2f→%.2f (Δ%+.2f) | SSIM: %.4f→%.4f (Δ%+.4f)",
                    detection_str, tools_str,
                    psnr_before, psnr_after, psnr_after - psnr_before,
                    ssim_before, ssim_after, ssim_after - ssim_before,
                )

                if do_save_images:
                    for win in ("brain", "subdural"):
                        fname = f"{case_label}_{art_type}_{sev}_restore_{win}.png"
                        plot_restoration_panel(
                            clean_recon, degraded, restored,
                            art_type, sev, detection_str, tools_str,
                            psnr_before, ssim_before, psnr_after, ssim_after,
                            case_label, win, out_dir / fname,
                        )

    # --- Summary ---
    summary_path = out_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # --- Detailed per-row table (only for small runs) ---
    if len(all_results) <= 100:
        logger.info("\n" + "=" * 80)
        logger.info("PER-CASE RESULTS")
        logger.info("=" * 80)
        header = f"{'Type':<20} {'Sev':<10} {'Detected':<40} {'Tools':<35} {'PSNR':>6} {'→':>2} {'PSNR':>6} {'ΔPSNR':>7} {'SSIM':>7} {'→':>2} {'SSIM':>7} {'ΔSSIM':>8}"
        logger.info(header)
        logger.info("-" * 160)
        for r in all_results:
            det_short = r["detected"][:38] if len(r["detected"]) > 38 else r["detected"]
            tools_short = r["tools"][:33] if len(r["tools"]) > 33 else r["tools"]
            logger.info(
                f"{r['artifact_type']:<20} {r['severity']:<10} {det_short:<40} {tools_short:<35} "
                f"{r['psnr_before']:>6.2f}  → {r['psnr_after']:>6.2f} {r['psnr_delta']:>+7.2f} "
                f"{r['ssim_before']:>7.4f}  → {r['ssim_after']:>7.4f} {r['ssim_delta']:>+8.4f}"
            )

    # --- Aggregate statistics ---
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE STATISTICS  [planner=%s]", args.planner)
    logger.info("=" * 80)

    acted = [r for r in all_results if not r["skipped"]]
    skipped = [r for r in all_results if r["skipped"]]
    improved = [r for r in acted if r["psnr_delta"] > 0]
    degraded = [r for r in acted if r["psnr_delta"] < 0]

    logger.info(f"Total experiments:     {len(all_results)}")
    logger.info(f"Skipped (do-no-harm):  {len(skipped)} ({100*len(skipped)/max(len(all_results),1):.1f}%)")
    logger.info(f"Acted on:              {len(acted)} ({100*len(acted)/max(len(all_results),1):.1f}%)")
    logger.info(f"  ├─ Improved (ΔPSNR>0): {len(improved)} ({100*len(improved)/max(len(acted),1):.1f}%)")
    logger.info(f"  └─ Degraded (ΔPSNR<0): {len(degraded)} ({100*len(degraded)/max(len(acted),1):.1f}%)")

    def _avg(lst: list[dict], key: str) -> float:
        vals = [r[key] for r in lst]
        return sum(vals) / len(vals) if vals else 0.0

    if acted:
        logger.info("\n--- Acted-on subset (tools were applied) ---")
        logger.info(f"  Avg PSNR before:  {_avg(acted, 'psnr_before'):.2f}")
        logger.info(f"  Avg PSNR after:   {_avg(acted, 'psnr_after'):.2f}")
        logger.info(f"  Avg ΔPSNR:        {_avg(acted, 'psnr_delta'):+.2f}")
        logger.info(f"  Avg SSIM before:  {_avg(acted, 'ssim_before'):.4f}")
        logger.info(f"  Avg SSIM after:   {_avg(acted, 'ssim_after'):.4f}")
        logger.info(f"  Avg ΔSSIM:        {_avg(acted, 'ssim_delta'):+.4f}")

    if skipped:
        logger.info("\n--- Skipped subset (no tools applied) ---")
        logger.info(f"  Avg PSNR (unchanged): {_avg(skipped, 'psnr_before'):.2f}")
        logger.info(f"  Avg SSIM (unchanged): {_avg(skipped, 'ssim_before'):.4f}")

    # --- Per artifact type × severity breakdown ---
    logger.info("\n--- Per artifact type × severity ---")
    header = f"{'Type':<20} {'Severity':<10} {'N':>5} {'Skip':>5} {'Act':>5} {'Impr':>5} {'AvgΔPSNR':>9} {'AvgΔSSIM':>10}"
    logger.info(header)
    logger.info("-" * 75)

    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["artifact_type"], r["severity"])].append(r)

    artifact_types_summary = [
        "ring", "motion", "beam_hardening", "scatter", "truncation",
        "low_dose", "sparse_view", "limited_angle", "focal_spot_blur",
    ]
    severities = ["mild", "moderate", "severe"]
    for art in artifact_types_summary:
        for sev in severities:
            g = groups.get((art, sev), [])
            if not g:
                continue
            n_skip = sum(1 for r in g if r["skipped"])
            n_act = len(g) - n_skip
            g_acted = [r for r in g if not r["skipped"]]
            n_impr = sum(1 for r in g_acted if r["psnr_delta"] > 0)
            avg_dpsnr = _avg(g_acted, "psnr_delta") if g_acted else 0.0
            avg_dssim = _avg(g_acted, "ssim_delta") if g_acted else 0.0
            logger.info(
                f"{art:<20} {sev:<10} {len(g):>5} {n_skip:>5} {n_act:>5} {n_impr:>5} {avg_dpsnr:>+9.2f} {avg_dssim:>+10.4f}"
            )

    # --- Per artifact type overall ---
    logger.info("\n--- Per artifact type (all severities) ---")
    header = f"{'Type':<20} {'N':>5} {'Skip':>5} {'Act':>5} {'Impr%':>6} {'AvgΔPSNR':>9} {'AvgΔSSIM':>10}"
    logger.info(header)
    logger.info("-" * 65)
    for art in artifact_types_summary:
        g = [r for r in all_results if r["artifact_type"] == art]
        if not g:
            continue
        n_skip = sum(1 for r in g if r["skipped"])
        g_acted = [r for r in g if not r["skipped"]]
        n_impr = sum(1 for r in g_acted if r["psnr_delta"] > 0)
        impr_pct = 100 * n_impr / max(len(g_acted), 1)
        avg_dpsnr = _avg(g_acted, "psnr_delta") if g_acted else 0.0
        avg_dssim = _avg(g_acted, "ssim_delta") if g_acted else 0.0
        logger.info(
            f"{art:<20} {len(g):>5} {n_skip:>5} {len(g_acted):>5} {impr_pct:>5.1f}% {avg_dpsnr:>+9.2f} {avg_dssim:>+10.4f}"
        )

    if use_llm:
        logger.info("\n--- LLM Usage ---")
        n_llm_calls = sum(1 for r in all_results if not r["skipped"])
        logger.info(f"  Total LLM planning calls: {n_llm_calls}")
        logger.info(f"  Skipped (rule fallback):   {len(all_results) - n_llm_calls}")

    logger.info("\n" + "=" * 80)
    logger.info("All results saved to %s", out_dir)
    logger.info("Summary JSON: %s", summary_path)


if __name__ == "__main__":
    main()
