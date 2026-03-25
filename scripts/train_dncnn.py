#!/usr/bin/env python
# ============================================================================
# 脚本职责: 在 CQ500 clean/degraded 数据对上快速训练 DnCNN
#
# 用法:
#   PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/train_dncnn.py \
#       --config configs/model/dncnn_train.yaml
#
# 参考: DnCNN (https://arxiv.org/abs/1608.03981)
# ============================================================================
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_dncnn")


def _collect_all_cases(processed_dir: str) -> list[str]:
    """Return sorted list of case directory names."""
    processed = Path(processed_dir)
    return sorted(d.name for d in processed.iterdir() if d.is_dir())


def _split_cases(
    all_cases: list[str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministic train/test split by case ID."""
    rng = random.Random(seed)
    shuffled = list(all_cases)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_cases = set(shuffled[:n_test])
    train_cases = [c for c in all_cases if c not in test_cases]
    return train_cases, sorted(test_cases)


class CTPatchDataset(Dataset):
    """Load clean/degraded pairs from CQ500 processed h5 files and extract patches."""

    def __init__(
        self,
        processed_dir: str,
        patch_size: int = 64,
        patches_per_image: int = 8,
        mask_idx: int = 0,
        max_cases: int | None = None,
        case_whitelist: list[str] | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.pairs: list[tuple[Path, Path]] = []

        processed = Path(processed_dir)
        case_dirs = sorted(d for d in processed.iterdir() if d.is_dir())

        if case_whitelist is not None:
            allowed = set(case_whitelist)
            case_dirs = [d for d in case_dirs if d.name in allowed]

        if max_cases:
            case_dirs = case_dirs[:max_cases]

        for case_dir in case_dirs:
            for series_dir in case_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                for slice_dir in series_dir.iterdir():
                    if not slice_dir.is_dir():
                        continue
                    gt_h5 = slice_dir / "gt.h5"
                    deg_h5 = slice_dir / f"{mask_idx}.h5"
                    if gt_h5.exists() and deg_h5.exists():
                        self.pairs.append((gt_h5, deg_h5))

        logger.info("Found %d clean/degraded pairs", len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx = idx // self.patches_per_image

        gt_path, deg_path = self.pairs[pair_idx]
        with h5py.File(str(gt_path), "r") as f:
            clean = f["image"][:].astype(np.float32)
        with h5py.File(str(deg_path), "r") as f:
            degraded = f["ma_CT"][:].astype(np.float32)

        h, w = clean.shape
        ps = self.patch_size
        y = random.randint(0, max(h - ps, 0))
        x = random.randint(0, max(w - ps, 0))
        clean_patch = clean[y:y+ps, x:x+ps]
        degraded_patch = degraded[y:y+ps, x:x+ps]

        degraded_patch = np.clip(degraded_patch, 0.0, 0.6)

        return (
            torch.from_numpy(degraded_patch).unsqueeze(0),
            torch.from_numpy(clean_patch).unsqueeze(0),
        )


def build_dncnn(depth: int = 17, channels: int = 64) -> nn.Sequential:
    """Standard DnCNN architecture."""
    layers: list[nn.Module] = [
        nn.Conv2d(1, channels, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
    ]
    for _ in range(depth - 2):
        layers.extend([
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ])
    layers.append(nn.Conv2d(channels, 1, 3, padding=1, bias=False))
    return nn.Sequential(*layers)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for structure preservation."""

    def __init__(self, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = self.kernel.to(x.device)  # type: ignore[union-attr]
        mu_x = torch.nn.functional.conv2d(x, k, padding=k.shape[-1] // 2)
        mu_y = torch.nn.functional.conv2d(y, k, padding=k.shape[-1] // 2)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sigma_x2 = torch.nn.functional.conv2d(x * x, k, padding=k.shape[-1] // 2) - mu_x2
        sigma_y2 = torch.nn.functional.conv2d(y * y, k, padding=k.shape[-1] // 2) - mu_y2
        sigma_xy = torch.nn.functional.conv2d(x * y, k, padding=k.shape[-1] // 2) - mu_xy

        ssim_map = ((2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)) / \
                   ((mu_x2 + mu_y2 + self.c1) * (sigma_x2 + sigma_y2 + self.c2))

        return 1.0 - ssim_map.mean()


def _eval_val(model: nn.Module, val_loader: DataLoader, device: str) -> tuple[float, float]:
    """Quick validation: return (avg_psnr, avg_ssim) on full images."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for degraded, clean in val_loader:
            degraded = degraded.to(device)
            noise_pred = model(degraded)
            denoised = (degraded - noise_pred).clamp(0.0, 0.6).cpu().numpy()
            clean_np = clean.numpy()
            for i in range(denoised.shape[0]):
                d = denoised[i, 0]
                c = clean_np[i, 0]
                dr = max(c.max() - c.min(), 1e-10)
                psnrs.append(peak_signal_noise_ratio(c, d, data_range=dr))
                ssims.append(structural_similarity(c, d, data_range=dr))
    return float(np.mean(psnrs)) if psnrs else 0.0, float(np.mean(ssims)) if ssims else 0.0


def train(cfg: dict[str, Any]) -> None:
    device = cfg.get("device", "cuda:0")
    epochs = cfg.get("epochs", 30)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-4)
    patch_size = cfg.get("patch_size", 64)
    patches_per_image = cfg.get("patches_per_image", 8)
    depth = cfg.get("depth", 17)
    channels = cfg.get("channels", 64)
    processed_dir = cfg["data"]["processed_dir"]
    max_train_cases = cfg.get("data", {}).get("max_train_cases", None)
    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    loss_weights = cfg.get("loss_weights", {"l1": 1.0, "ssim": 0.3})
    test_ratio = cfg.get("data", {}).get("test_ratio", 0.2)

    # --- Train/test split ---
    all_cases = _collect_all_cases(processed_dir)
    train_cases, test_cases = _split_cases(all_cases, test_ratio=test_ratio)
    logger.info("Split: %d train cases, %d test cases (ratio=%.0f%%)",
                len(train_cases), len(test_cases), test_ratio * 100)

    split_info = {"train_cases": train_cases, "test_cases": test_cases}
    import json
    split_path = save_dir / "train_test_split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Saved train/test split to %s", split_path)

    # --- Datasets ---
    train_dataset = CTPatchDataset(
        processed_dir=processed_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        max_cases=max_train_cases,
        case_whitelist=train_cases,
    )
    val_dataset = CTPatchDataset(
        processed_dir=processed_dir,
        patch_size=patch_size,
        patches_per_image=2,
        case_whitelist=test_cases[:20],
    )

    if len(train_dataset) == 0:
        logger.error("No training data found. Check processed_dir.")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = build_dncnn(depth=depth, channels=channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("DnCNN: depth=%d, channels=%d, params=%.2fM", depth, channels, total_params / 1e6)
    logger.info("Train pairs: %d, Val pairs: %d", len(train_dataset.pairs), len(val_dataset.pairs))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    l1_loss_fn = nn.L1Loss()
    ssim_loss_fn = SSIMLoss().to(device)

    best_val_ssim = 0.0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0

        for degraded, clean in train_loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            noise_pred = model(degraded)
            denoised = degraded - noise_pred

            loss_l1 = l1_loss_fn(denoised, clean)
            loss_ssim = ssim_loss_fn(denoised, clean)
            loss = loss_weights["l1"] * loss_l1 + loss_weights["ssim"] * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batch, 1)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_psnr, val_ssim = _eval_val(model, val_loader, device)
            elapsed = time.time() - t_start
            logger.info(
                "Epoch %d/%d  loss=%.6f  val_PSNR=%.2f  val_SSIM=%.4f  lr=%.2e  elapsed=%.0fs",
                epoch + 1, epochs, avg_loss, val_psnr, val_ssim,
                scheduler.get_last_lr()[0], elapsed,
            )

            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                save_path = save_dir / "dncnn_ct.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "config": cfg,
                }, str(save_path))
                logger.info("  => New best val SSIM=%.4f, saved checkpoint", val_ssim)

    elapsed = time.time() - t_start
    logger.info("Training complete. %.0fs total. Best val SSIM=%.4f", elapsed, best_val_ssim)
    logger.info("Weights saved to %s", save_dir / "dncnn_ct.pth")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Train DnCNN on CQ500 CT data")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Limit training cases")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.max_cases:
        cfg.setdefault("data", {})["max_train_cases"] = args.max_cases

    train(cfg)


if __name__ == "__main__":
    main()
