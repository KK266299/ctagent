# ============================================================================
# 模块职责: DnCNN Learned Denoiser — 基于 CNN 的 CT 图像去噪工具
#   - 标准 DnCNN 残差学习 (predict noise, subtract)
#   - 支持加载预训练权重
#   - 无权重时 fallback 到 denoise_tv
# 参考: DnCNN (https://arxiv.org/abs/1608.03981)
#       RED-CNN (https://arxiv.org/abs/1702.00288)
# ============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.base import BaseTool, ToolMeta, ToolResult
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "dncnn_ct.pth"

_model_cache: dict[str, Any] = {}


def _build_dncnn(depth: int = 17, channels: int = 64) -> Any:
    """Build a DnCNN model (residual learning for denoising)."""
    import torch
    import torch.nn as nn

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

    model = nn.Sequential(*layers)
    return model


def _load_model(weights_path: str | Path, device: str = "cpu") -> Any:
    """Load DnCNN with caching."""
    import torch

    key = f"{weights_path}_{device}"
    if key in _model_cache:
        return _model_cache[key]

    model = _build_dncnn()
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    _model_cache[key] = model
    logger.info("Loaded DnCNN weights from %s (device=%s)", weights_path, device)
    return model


@ToolRegistry.register
class DnCNNDenoise(BaseTool):
    """DnCNN — CNN-based CT image denoiser with residual learning.

    Predicts noise residual and subtracts from input.
    Falls back to TV denoising if no trained weights are available.
    """

    @property
    def name(self) -> str:
        return "denoise_dncnn"

    @property
    def description(self) -> str:
        return (
            "DnCNN deep learning denoiser: residual CNN trained on CT data. "
            "Best PSNR/SSIM among all denoisers when trained weights are available. "
            "Falls back to TV denoising if no weights found."
        )

    @property
    def meta(self) -> ToolMeta:
        return ToolMeta(
            category="denoise",
            suitable_for=["noise", "artifact"],
            expected_cost="medium",
            expected_safety="safe",
            params_schema={
                "weights_path": {
                    "type": "str",
                    "default": str(_DEFAULT_WEIGHTS_PATH),
                    "description": "Path to trained DnCNN weights (.pth)",
                },
                "device": {
                    "type": "str",
                    "default": "auto",
                    "description": "auto = use CUDA if available, else CPU",
                },
                "blend": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 1.0,
                    "description": "Blending factor: 1.0 = full DnCNN, 0.0 = keep input",
                },
            },
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        weights_path = Path(kwargs.get("weights_path", _DEFAULT_WEIGHTS_PATH))
        device_arg = str(kwargs.get("device", "auto"))
        if device_arg == "auto":
            import torch as _torch
            device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        else:
            device = device_arg
        blend = float(kwargs.get("blend", 1.0))

        if not weights_path.exists():
            logger.warning("DnCNN weights not found at %s, falling back to TV denoise", weights_path)
            return self._fallback_tv(image)

        try:
            import torch

            model = _load_model(weights_path, device)
            img_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                noise_pred = model(img_tensor)
                denoised = img_tensor - noise_pred

            result_np = denoised.squeeze().cpu().numpy()

            if blend < 1.0:
                result_np = blend * result_np + (1.0 - blend) * image.astype(np.float32)

            result_np = np.clip(result_np, 0.0, max(image.max(), 0.6))

            return ToolResult(
                image=result_np.astype(np.float32),
                tool_name=self.name,
                metadata={"backend": "dncnn", "device": device, "blend": blend},
            )

        except Exception as e:
            logger.warning("DnCNN inference failed (%s), falling back to TV denoise", e)
            return self._fallback_tv(image)

    @staticmethod
    def _fallback_tv(image: np.ndarray) -> ToolResult:
        from skimage.restoration import denoise_tv_chambolle
        result = denoise_tv_chambolle(image.astype(np.float64), weight=0.05)
        return ToolResult(
            image=result.astype(np.float32),
            tool_name="denoise_dncnn",
            metadata={"backend": "fallback_tv"},
        )
