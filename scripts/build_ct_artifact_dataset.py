#!/usr/bin/env python
# ============================================================================
# 脚本职责: CT 通用伪影数据集批量构建入口
#   读取配置 → 加载几何/物理参数 → 扫描 CT slices → 生成 5 类伪影 → 保存 HDF5
#   支持输入源: CQ500 DICOM / DeepLesion PNG / 通用 DICOM 目录
#   输出结构:
#       <output>/<artifact_type>/<severity>/<case_id>/gt.h5
#       <output>/<artifact_type>/<severity>/<case_id>/<artifact_type>_<severity>.h5
# ============================================================================
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from dataset.mar import create_artifact_simulator
from dataset.mar.ct_geometry import CTGeometry, CTGeometryConfig
from dataset.mar.physics_params import PhysicsConfig, PhysicsParams
from scripts.build_mar_dataset import SOURCE_DISPATCH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_ct_artifact_dataset")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CT artifact simulation dataset")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--max-images", type=int, default=None, help="Override max slices")
    parser.add_argument(
        "--artifact-types",
        type=str,
        default=None,
        help="Comma separated artifact types, e.g. ring,motion,scatter",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default=None,
        help="Comma separated severities, e.g. mild,moderate,severe",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only scan and count")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_type = cfg["input"].get("source_type", "cq500")
    if source_type not in SOURCE_DISPATCH:
        logger.error("Unknown source_type: %s", source_type)
        sys.exit(1)

    geo_cfg = CTGeometryConfig(**cfg.get("geometry", {}))
    target_size = geo_cfg.image_size
    batch_cfg = cfg.get("batch", {})
    max_imgs = args.max_images or batch_cfg.get("max_images")

    artifacts_cfg = cfg.get("artifacts", {})
    artifact_types = artifacts_cfg.get("enabled_types", ["ring"])
    if args.artifact_types:
        artifact_types = [x.strip() for x in args.artifact_types.split(",") if x.strip()]

    severities = artifacts_cfg.get("severities", ["moderate"])
    if args.severities:
        severities = [x.strip() for x in args.severities.split(",") if x.strip()]

    slice_iter = SOURCE_DISPATCH[source_type]
    tasks = []
    logger.info("Scanning slices ...")
    for hu_image, case_id in slice_iter(cfg, max_imgs, target_size):
        tasks.append((hu_image, case_id))
    logger.info("Collected %d slices", len(tasks))

    if args.dry_run:
        total = len(tasks) * len(artifact_types) * len(severities)
        logger.info(
            "Dry run: %d slices × %d artifact_types × %d severities = %d outputs",
            len(tasks),
            len(artifact_types),
            len(severities),
            total,
        )
        return

    if not tasks:
        logger.warning("No slices found, exiting.")
        return

    output_dir = Path(cfg["output"]["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    minimal = cfg.get("output", {}).get("minimal", True)

    phy_kwargs = {**cfg.get("physics", {})}
    phy_kwargs["mat_dir"] = cfg["input"]["mat_dir"]
    mat_files = {}
    for key in ["mat_water", "mat_bone", "mat_metals", "mat_spectrum", "mat_masks"]:
        if key in cfg["input"]:
            mat_files[key] = cfg["input"][key]

    geometry = CTGeometry(geo_cfg)
    phy_cfg = PhysicsConfig(**phy_kwargs, **mat_files)
    physics = PhysicsParams(phy_cfg)
    physics.load()

    manifest = []
    global_seed = int(batch_cfg.get("seed", 42))
    t0 = time.time()
    num_done = 0

    for idx, (hu_image, case_id) in enumerate(tasks):
        case_id_safe = case_id.replace(" ", "_")
        for artifact_offset, artifact_type in enumerate(artifact_types):
            for severity_offset, severity in enumerate(severities):
                seed = global_seed + idx * 100 + artifact_offset * 10 + severity_offset
                simulator = create_artifact_simulator(
                    artifact_type,
                    geometry,
                    physics,
                    seed=seed,
                    minimal=minimal,
                )
                try:
                    result = simulator.simulate(hu_image, severity=severity)
                    out_sub = output_dir / artifact_type / severity / case_id_safe
                    simulator.save_h5(result, out_sub, minimal=minimal)
                    manifest.append({
                        "case_id": case_id,
                        "artifact_type": artifact_type,
                        "severity": severity,
                        "output_dir": str(out_sub),
                        "result_file": f"{artifact_type}_{severity}.h5",
                    })
                    num_done += 1
                except Exception as e:
                    logger.warning(
                        "Failed case=%s artifact=%s severity=%s: %s",
                        case_id,
                        artifact_type,
                        severity,
                        e,
                    )
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = num_done / max(elapsed, 1e-6)
            logger.info("Progress: %d/%d slices, %d outputs (%.1f output/min)", idx + 1, len(tasks), num_done, rate * 60)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(
        "Done! %d outputs from %d slices in %.1f min",
        num_done,
        len(tasks),
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
