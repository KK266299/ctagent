#!/usr/bin/env python
# ============================================================================
# 脚本职责: 可视化 clean / degraded / restored 对比图
#   对指定 case 的切片生成 4-panel 对比 PNG:
#     [clean | degraded | restored_rule | restored_llm]
#   同时输出 PSNR/SSIM 到图上
#
# 用法:
#   PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 bash -c '
#   eval "$(conda shell.bash hook 2>/dev/null)" && conda activate llamafactory && \
#   python -u scripts/visualize_restoration.py \
#       --config configs/experiment/cq500_iqa_eval.yaml \
#       --output-dir /home/liuxinyao/project/ctagent/try_output/restoration_vis \
#       --max-cases 10 --max-slices 2'
# ============================================================================
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vis_restore")


MU_WATER = 0.192


def hu_window_to_mu(wl_hu: float, ww_hu: float) -> tuple[float, float]:
    """Convert HU window (center, width) to μ-value range [lo, hi]."""
    hu_lo = wl_hu - ww_hu / 2.0
    hu_hi = wl_hu + ww_hu / 2.0
    mu_lo = MU_WATER * (1.0 + hu_lo / 1000.0)
    mu_hi = MU_WATER * (1.0 + hu_hi / 1000.0)
    return mu_lo, mu_hi


def mu_to_display(
    img: np.ndarray,
    window: str = "brain",
) -> np.ndarray:
    """Convert μ-value image to [0,1] display range using standard CT windows.

    Windows (HU):
      brain:    WL=40,  WW=80   → soft tissue contrast
      subdural: WL=75,  WW=215  → wider range, good for hemorrhage
      bone:     WL=500, WW=2000 → bone detail
      stroke:   WL=40,  WW=40   → acute hemorrhage
    """
    windows = {
        "brain":    (40, 80),
        "subdural": (75, 215),
        "bone":     (500, 2000),
        "stroke":   (40, 40),
    }
    wl_hu, ww_hu = windows.get(window, (40, 80))
    lo, hi = hu_window_to_mu(wl_hu, ww_hu)
    out = (img - lo) / max(hi - lo, 1e-10)
    return np.clip(out, 0.0, 1.0)


def restore_single(
    degraded_mu: np.ndarray,
    clean_mu: np.ndarray,
    planner: object,
    detector: object,
    restoration_tool: object,
    report: object = None,
) -> tuple[np.ndarray, list[str]]:
    """Run restoration and return (restored_image, tool_names)."""
    if report is None:
        report = detector.detect(degraded_mu)
    plan = planner.plan(report)
    if plan is None or len(plan) == 0:
        return degraded_mu.copy(), []
    steps = [{"tool_name": s.tool_name, "params": s.params} for s in plan.steps]
    result = restoration_tool.apply_chain(degraded_mu, steps, reference=clean_mu)
    restored = result.get("_restored_image", degraded_mu)
    tool_names = [s.tool_name for s in plan.steps]
    return restored, tool_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize restoration results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-slices", type=int, default=2)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Imports
    from eval.cq500_manifest import build_eval_manifest, SOPIndex
    from src.degradations.detector import DegradationDetector
    from src.planner.rule_planner import RuleBasedPlanner
    from src.tools.mcp_style.restoration_tool import RestorationTool

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

    data_cfg = cfg.get("data", {})
    detector_cfg = cfg.get("detector", {})

    sop_index = None
    bhx_annotations = None
    bhx_csv = data_cfg.get("bhx_csv")
    sop_index_path = data_cfg.get("sop_index")
    if bhx_csv and sop_index_path:
        if Path(bhx_csv).exists() and Path(sop_index_path).exists():
            from eval.bhx_loader import BHXAnnotations
            bhx_annotations = BHXAnnotations(bhx_csv)
            sop_index = SOPIndex(sop_index_path)

    cases = build_eval_manifest(
        processed_dir=data_cfg["processed_dir"],
        max_slices_per_case=args.max_slices,
        mask_idx=data_cfg.get("mask_idx", 0),
        sop_index=sop_index,
        bhx_annotations=bhx_annotations,
    )

    if args.max_cases:
        cases = cases[:args.max_cases]

    detector = DegradationDetector(detector_cfg)
    rule_planner = RuleBasedPlanner(max_chain=cfg.get("planner", {}).get("max_chain", 4))
    restoration_tool = RestorationTool()

    # Optional LLM planner
    agent_planner = None
    try:
        from llm.api_client import LLMConfig, create_client
        from llm.planner_caller import PlannerCaller
        from src.planner.agent_based import AgentBasedPlanner

        llm_cfg = cfg.get("llm", {})
        llm_config = LLMConfig(
            provider=llm_cfg.get("provider", "openai"),
            model=args.model or llm_cfg.get("model", "qwen/qwen-2.5-vl-72b-instruct"),
            base_url=args.base_url if args.base_url else llm_cfg.get("base_url"),
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
        logger.info("LLM planner ready: %s", llm_config.model)
    except Exception as e:
        logger.warning("LLM planner not available: %s", e)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for se in case.slices:
            try:
                with h5py.File(str(se.gt_h5), "r") as f:
                    clean_mu = f["image"][:].astype(np.float64)
                with h5py.File(str(se.degraded_h5), "r") as f:
                    degraded_mu = f["ma_CT"][:].astype(np.float64)
            except Exception as e:
                logger.warning("Skip %s/%s: %s", case.case_id, se.slice_dir.name, e)
                continue

            data_range = float(max(clean_mu.max() - clean_mu.min(), 1e-10))
            report = detector.detect(degraded_mu)

            # Rule restoration
            restored_rule, tools_rule = restore_single(
                degraded_mu, clean_mu, rule_planner, detector, restoration_tool, report
            )

            # LLM restoration (if available)
            restored_llm, tools_llm = None, []
            if agent_planner is not None:
                try:
                    restored_llm, tools_llm = restore_single(
                        degraded_mu, clean_mu, agent_planner, detector, restoration_tool, report
                    )
                except Exception as e:
                    logger.warning("LLM restore failed: %s", e)

            # Metrics
            deg_psnr = psnr_fn(clean_mu, degraded_mu, data_range=data_range)
            deg_ssim = ssim_fn(clean_mu, degraded_mu, data_range=data_range)
            rule_psnr = psnr_fn(clean_mu, restored_rule, data_range=data_range)
            rule_ssim = ssim_fn(clean_mu, restored_rule, data_range=data_range)

            n_panels = 3
            if restored_llm is not None:
                llm_psnr = psnr_fn(clean_mu, restored_llm, data_range=data_range)
                llm_ssim = ssim_fn(clean_mu, restored_llm, data_range=data_range)
                n_panels = 4

            # Plot: 2 rows (brain window + bone window) × n_panels columns
            for win_name in ["brain", "subdural"]:
                fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

                clean_disp = mu_to_display(clean_mu, window=win_name)
                deg_disp = mu_to_display(np.clip(degraded_mu, -0.2, 0.6), window=win_name)
                rule_disp = mu_to_display(restored_rule, window=win_name)

                axes[0].imshow(clean_disp, cmap="gray")
                axes[0].set_title("Clean (GT)", fontsize=11)
                axes[0].axis("off")

                axes[1].imshow(deg_disp, cmap="gray")
                axes[1].set_title(f"Degraded\nPSNR={deg_psnr:.1f} SSIM={deg_ssim:.3f}", fontsize=10)
                axes[1].axis("off")

                axes[2].imshow(rule_disp, cmap="gray")
                tools_str = "→".join(tools_rule[:3]) if tools_rule else "none"
                axes[2].set_title(
                    f"Rule: {tools_str}\nPSNR={rule_psnr:.1f} SSIM={rule_ssim:.3f}", fontsize=10
                )
                axes[2].axis("off")

                if n_panels == 4 and restored_llm is not None:
                    llm_disp = mu_to_display(restored_llm, window=win_name)
                    axes[3].imshow(llm_disp, cmap="gray")
                    tools_str_llm = "→".join(tools_llm[:3]) if tools_llm else "none"
                    axes[3].set_title(
                        f"LLM: {tools_str_llm}\nPSNR={llm_psnr:.1f} SSIM={llm_ssim:.3f}", fontsize=10
                    )
                    axes[3].axis("off")

                fig.suptitle(
                    f"{case.case_id} / {se.slice_dir.name}  [{win_name} window]",
                    fontsize=12, y=1.02,
                )
                plt.tight_layout()

                fname = f"{case.case_id}_{se.slice_dir.name}_{win_name}.png"
                fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)

            logger.info("Saved %s/%s (brain + subdural)", case.case_id, se.slice_dir.name)

    logger.info("All visualizations saved to %s", out_dir)


if __name__ == "__main__":
    main()
