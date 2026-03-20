# ============================================================================
# 模块职责: MCP-style 工具单元测试
# ============================================================================
import numpy as np

from src.tools.mcp_style.analysis_tool import AnalysisTool
from src.tools.mcp_style.perception_tool import PerceptionTool
from src.tools.mcp_style.statistics_tool import StatisticsTool
from src.tools.mcp_style.restoration_tool import RestorationTool


def _make_test_image(noise_level: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.random((64, 64)).astype(np.float32) * 0.5 + 0.25
    if noise_level > 0:
        img += rng.normal(0, noise_level, img.shape).astype(np.float32)
    return img


class TestAnalysisTool:
    def test_returns_dict(self):
        tool = AnalysisTool()
        result = tool(_make_test_image())
        assert isinstance(result, dict)
        assert result["tool"] == "ct_degradation_analysis"

    def test_detects_noise(self):
        tool = AnalysisTool()
        result = tool(_make_test_image(noise_level=0.5))
        assert result["num_degradations"] > 0

    def test_clean_image(self):
        tool = AnalysisTool()
        # 非常平滑的图像
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        result = tool(img)
        assert result["primary_degradation"] == "none"


class TestPerceptionTool:
    def test_no_reference(self):
        tool = PerceptionTool()
        result = tool(_make_test_image())
        assert "no_reference" in result
        assert "sharpness" in result["no_reference"]
        assert "quality_grade" in result

    def test_with_reference(self):
        tool = PerceptionTool()
        img = _make_test_image()
        ref = _make_test_image()
        result = tool(img, reference=ref)
        assert "full_reference" in result
        assert "psnr" in result["full_reference"]


class TestStatisticsTool:
    def test_global_stats(self):
        tool = StatisticsTool()
        result = tool(_make_test_image())
        assert "global" in result
        assert "mean" in result["global"]
        assert "histogram_features" in result

    def test_roi_stats(self):
        tool = StatisticsTool()
        result = tool(_make_test_image(), roi_center=(32, 32), roi_size=16)
        assert "roi" in result


class TestRestorationTool:
    def test_restore_success(self):
        # 确保经典工具已注册
        import src.tools.classical  # noqa: F401
        tool = RestorationTool()
        result = tool(_make_test_image(noise_level=0.1), tool_name="denoise_gaussian")
        assert result["success"]
        assert "quality_before" in result
        assert "quality_after" in result

    def test_restore_unknown_tool(self):
        tool = RestorationTool()
        result = tool(_make_test_image(), tool_name="nonexistent_tool")
        assert not result["success"]
        assert "error" in result

    def test_list_available(self):
        import src.tools.classical  # noqa: F401
        tool = RestorationTool()
        tools = tool.list_available_tools()
        assert "denoise_gaussian" in tools
