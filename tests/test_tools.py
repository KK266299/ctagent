# ============================================================================
# 模块职责: 工具模块单元测试
# ============================================================================
import numpy as np

from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry


def test_registry_lists_tools():
    """注册表应包含已注册工具。"""
    # 触发注册
    import src.tools.classical  # noqa: F401
    tools = ToolRegistry.list_tools()
    assert "denoise_nlm" in tools
    assert "sharpen_usm" in tools
    assert "histogram_clahe" in tools


def test_registry_create():
    """应能通过名称创建工具实例。"""
    import src.tools.classical  # noqa: F401
    tool = ToolRegistry.create("denoise_gaussian")
    assert isinstance(tool, BaseTool)
    assert tool.name == "denoise_gaussian"


def test_gaussian_denoise():
    """高斯去噪应返回有效结果。"""
    import src.tools.classical  # noqa: F401
    tool = ToolRegistry.create("denoise_gaussian")
    img = np.random.rand(64, 64).astype(np.float32)
    result = tool.run(img, sigma=1.0)
    assert isinstance(result, ToolResult)
    assert result.success
    assert result.image.shape == img.shape
