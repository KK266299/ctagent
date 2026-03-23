#!/usr/bin/env python
# ============================================================================
# 脚本职责: MAR 退化数据集批量构建入口
#   读取配置 → 加载物理参数 → 遍历 CT 图像 → 仿真 → 保存 HDF5
#   支持输入源: CQ500 DICOM / DeepLesion PNG / 通用 DICOM 目录
#   支持多进程 GPU 并行 (--workers N)
# 参考: ADN — prepare_deep_lesion.m / MAR_SynCode — prepare_deep_lesion.py
# 用法:
#   # CQ500 DICOM 模式 (4 进程并行)
#   PYTHONPATH=. python scripts/build_mar_dataset.py \
#       --config configs/data/mar_simulation_cq500.yaml --workers 4
#
#   # DeepLesion PNG 模式 (向后兼容, 单进程)
#   PYTHONPATH=. python scripts/build_mar_dataset.py \
#       --config configs/data/mar_simulation.yaml
# ============================================================================
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_mar_dataset")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 输入源: slice 迭代器
# ---------------------------------------------------------------------------

def iter_deeplesion_slices(cfg: dict, max_images: int | None, target_size: int):
    """DeepLesion PNG 模式: 递归扫描 PNG 文件。"""
    from dataset.mar.tissue_decompose import load_deeplesion_png

    image_dir = Path(cfg["input"]["ct_image_dir"])
    if not image_dir.exists():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    pngs = sorted(image_dir.rglob("*.png"))
    if max_images is not None:
        pngs = pngs[:max_images]
    logger.info("DeepLesion mode: %d PNG files", len(pngs))

    for png_path in pngs:
        hu = load_deeplesion_png(str(png_path), target_size)
        rel = png_path.relative_to(image_dir)
        case_id = str(rel.with_suffix(""))
        yield hu, case_id


def iter_cq500_slices(cfg: dict, max_images: int | None, target_size: int):
    """CQ500 DICOM 模式: 按 series 扫描并逐 slice 输出。"""
    from dataset.mar.cq500_reader import scan_cq500, read_dicom_hu

    input_cfg = cfg["input"]
    cq500_cfg = cfg.get("cq500", {})

    series_keywords = cq500_cfg.get("series_keywords")
    min_slices = cq500_cfg.get("min_slices", 20)
    stride = cq500_cfg.get("slice_stride", 1)
    max_patients = cq500_cfg.get("max_patients")

    series_list = scan_cq500(
        input_cfg["ct_image_dir"],
        series_keywords=series_keywords,
        min_slices=min_slices,
        max_patients=max_patients,
    )

    count = 0
    for sr in series_list:
        selected = sr.dcm_paths[::stride]
        for dcm_path in selected:
            if max_images is not None and count >= max_images:
                return
            try:
                hu = read_dicom_hu(dcm_path, target_size)
                case_id = f"{sr.case_id}/{dcm_path.stem}"
                yield hu, case_id
                count += 1
            except Exception as e:
                logger.warning("Skip %s: %s", dcm_path.name, e)


def iter_dicom_slices(cfg: dict, max_images: int | None, target_size: int):
    """通用 DICOM 模式: 递归扫描所有 .dcm。"""
    from dataset.mar.cq500_reader import read_dicom_hu

    image_dir = Path(cfg["input"]["ct_image_dir"])
    if not image_dir.exists():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    dcm_paths = sorted(image_dir.rglob("*.dcm"))
    if max_images is not None:
        dcm_paths = dcm_paths[:max_images]
    logger.info("DICOM mode: %d files", len(dcm_paths))

    for dcm_path in dcm_paths:
        try:
            hu = read_dicom_hu(dcm_path, target_size)
            rel = dcm_path.relative_to(image_dir)
            case_id = str(rel.with_suffix("")).replace(" ", "_")
            yield hu, case_id
        except Exception as e:
            logger.warning("Skip %s: %s", dcm_path.name, e)


SOURCE_DISPATCH = {
    "deeplesion": iter_deeplesion_slices,
    "cq500": iter_cq500_slices,
    "dicom": iter_dicom_slices,
}


# ---------------------------------------------------------------------------
# 单个任务项 (用于并行)
# ---------------------------------------------------------------------------

def _worker_init(cfg_dict: dict, max_masks: int | None):
    """每个 worker 进程的初始化: 创建独立的 GPU context + simulator。"""
    import os
    from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig
    from dataset.mar.physics_params import PhysicsConfig, PhysicsParams
    from dataset.mar.mar_simulator import MARSimulator

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    pid = os.getpid()

    geo_cfg = CTGeometryConfig(**cfg_dict.get("geometry", {}))
    geometry = CTGeometry(geo_cfg)

    phy_kwargs = {**cfg_dict.get("physics", {})}
    phy_kwargs["mat_dir"] = cfg_dict["input"]["mat_dir"]
    mat_files = {}
    for key in ["mat_water", "mat_bone", "mat_metals", "mat_spectrum", "mat_masks"]:
        if key in cfg_dict["input"]:
            mat_files[key] = cfg_dict["input"][key]
    phy_cfg = PhysicsConfig(**phy_kwargs, **mat_files)
    physics = PhysicsParams(phy_cfg)
    physics.load()

    masks_path = cfg_dict["input"].get("metal_masks_path")
    metal_masks = physics.load_metal_masks(masks_path)
    if max_masks is not None:
        metal_masks = metal_masks[:max_masks]

    batch_cfg = cfg_dict.get("batch", {})
    seed = batch_cfg.get("seed", 42) + pid
    minimal = cfg_dict.get("output", {}).get("minimal", False)
    simulator = MARSimulator(geometry, physics, seed=seed, minimal=minimal)

    global _g_simulator, _g_metal_masks, _g_minimal, _g_output_dir
    _g_simulator = simulator
    _g_metal_masks = metal_masks
    _g_minimal = minimal
    _g_output_dir = Path(cfg_dict["output"]["data_dir"])

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [W-{pid}] %(message)s",
        datefmt="%H:%M:%S",
    )


def _worker_process_slice(task: tuple[int, np.ndarray, str]) -> dict | None:
    """Worker 处理单个 slice。"""
    idx, hu_image, case_id = task
    t_start = time.time()
    try:
        result = _g_simulator.simulate(hu_image, _g_metal_masks)
        case_id_safe = case_id.replace(" ", "_")
        out_sub = _g_output_dir / case_id_safe
        _g_simulator.save_h5(result, out_sub, minimal=_g_minimal)
        elapsed = round(time.time() - t_start, 2)
        if idx % 50 == 0:
            logging.info("[%d] %s  (%.2fs)", idx, case_id, elapsed)
        return {
            "index": idx,
            "case_id": case_id,
            "output_dir": str(out_sub),
            "num_masks": len(result.mask_results),
            "time_sec": elapsed,
        }
    except Exception as e:
        import traceback
        logging.error("[%d] Failed: %s — %s\n%s", idx, case_id, e, traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build MAR simulation dataset")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Override max slices (for testing)")
    parser.add_argument("--max-masks", type=int, default=None,
                        help="Override max masks per slice")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only scan and count, don't simulate")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from dataset.mar.ct_geometry import CTGeometryConfig

    # --- 确定输入源 ---
    source_type = cfg["input"].get("source_type", "deeplesion")
    if source_type not in SOURCE_DISPATCH:
        logger.error("Unknown source_type: %s (valid: %s)", source_type, list(SOURCE_DISPATCH.keys()))
        sys.exit(1)

    geo_cfg = CTGeometryConfig(**cfg.get("geometry", {}))
    target_size = geo_cfg.image_size

    # --- 获取 slice 迭代器 ---
    batch_cfg = cfg.get("batch", {})
    max_masks = args.max_masks or batch_cfg.get("max_masks_per_image")
    max_imgs = args.max_images or batch_cfg.get("max_images")
    slice_iter = SOURCE_DISPATCH[source_type]

    if args.dry_run:
        count = 0
        for _hu, _cid in slice_iter(cfg, max_imgs, target_size):
            count += 1
        n_masks = max_masks if max_masks else 100
        logger.info("Dry run: %d slices × %d masks = %d output groups",
                     count, n_masks, count * n_masks)
        return

    # --- 收集全部 slice 任务 ---
    logger.info("Scanning slices ...")
    tasks: list[tuple[int, np.ndarray, str]] = []
    for hu_image, case_id in slice_iter(cfg, max_imgs, target_size):
        tasks.append((len(tasks), hu_image, case_id))
    logger.info("Collected %d slices", len(tasks))

    if not tasks:
        logger.warning("No slices found, exiting.")
        return

    output_dir = Path(cfg["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    minimal = cfg.get("output", {}).get("minimal", False)
    num_workers = args.workers

    t0 = time.time()

    if num_workers <= 1:
        # --- 单进程模式 (向后兼容) ---
        from dataset.mar.ct_geometry import CTGeometry
        from dataset.mar.physics_params import PhysicsConfig, PhysicsParams
        from dataset.mar.mar_simulator import MARSimulator

        logger.info("Single-process mode (minimal=%s)", minimal)

        geo = CTGeometry(geo_cfg)

        phy_kwargs = {**cfg.get("physics", {})}
        phy_kwargs["mat_dir"] = cfg["input"]["mat_dir"]
        mat_files = {}
        for key in ["mat_water", "mat_bone", "mat_metals", "mat_spectrum", "mat_masks"]:
            if key in cfg["input"]:
                mat_files[key] = cfg["input"][key]
        phy_cfg = PhysicsConfig(**phy_kwargs, **mat_files)
        physics = PhysicsParams(phy_cfg)
        physics.load()

        masks_path = cfg["input"].get("metal_masks_path")
        metal_masks = physics.load_metal_masks(masks_path)
        if max_masks is not None:
            metal_masks = metal_masks[:max_masks]
        logger.info("Using %d metal masks", len(metal_masks))

        seed = batch_cfg.get("seed", 42)
        simulator = MARSimulator(geo, physics, seed=seed, minimal=minimal)

        manifest = []
        for idx, hu_image, case_id in tasks:
            t_start = time.time()
            try:
                result = simulator.simulate(hu_image, metal_masks)
                case_id_safe = case_id.replace(" ", "_")
                out_sub = output_dir / case_id_safe
                simulator.save_h5(result, out_sub, minimal=minimal)
                record = {
                    "index": idx,
                    "case_id": case_id,
                    "output_dir": str(out_sub),
                    "num_masks": len(result.mask_results),
                    "time_sec": round(time.time() - t_start, 2),
                }
                manifest.append(record)
            except Exception as e:
                import traceback
                logger.error("[%d] Failed: %s — %s\n%s", idx, case_id, e, traceback.format_exc())

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                logger.info("Progress: %d/%d (%.1f slice/min)", idx + 1, len(tasks), rate * 60)

    else:
        # --- 多进程并行模式 ---
        logger.info("Parallel mode: %d workers (minimal=%s)", num_workers, minimal)
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(cfg, max_masks),
        ) as pool:
            results = pool.map(_worker_process_slice, tasks, chunksize=4)

        manifest = [r for r in results if r is not None]
        manifest.sort(key=lambda r: r["index"])
        failed = sum(1 for r in results if r is None)
        if failed:
            logger.warning("%d slices failed", failed)

    # --- 保存 manifest ---
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(
        "Done! %d slices processed in %.1f min (%.1f slice/min)",
        len(manifest), elapsed / 60, len(manifest) / max(elapsed, 1) * 60,
    )


if __name__ == "__main__":
    main()
