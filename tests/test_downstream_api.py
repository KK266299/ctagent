# ============================================================================
# 模块职责: 闭源 API 相关模块单元测试
# ============================================================================
import json

from src.downstream.prompt_builder import PromptBuilder
from src.downstream.response_parser import ResponseParser


class TestPromptBuilder:
    def test_direct_diagnosis_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_direct_diagnosis_prompt()
        assert "CT image" in prompt
        assert "diagnostic" in prompt.lower()

    def test_direct_diagnosis_with_context(self):
        builder = PromptBuilder()
        prompt = builder.build_direct_diagnosis_prompt(context="degraded_ct")
        assert "degraded_ct" in prompt

    def test_tool_augmented_prompt(self):
        builder = PromptBuilder()
        tool_results = {
            "noise_analysis": {"noise_level": 25.3, "type": "gaussian"},
            "quality_score": {"sharpness": 45.0},
        }
        prompt = builder.build_tool_augmented_prompt(tool_results)
        assert "noise_analysis" in prompt
        assert "25.3" in prompt

    def test_tool_descriptions(self):
        builder = PromptBuilder()
        tools = {"denoise": "Remove noise", "sharpen": "Sharpen edges"}
        text = builder.build_tool_descriptions(tools)
        assert "denoise" in text
        assert "Remove noise" in text

    def test_system_prompt_exists(self):
        builder = PromptBuilder()
        assert len(builder.system_prompt) > 0
        assert "JSON" in builder.system_prompt


class TestResponseParser:
    def test_parse_json_response(self):
        parser = ResponseParser()
        response = json.dumps({
            "findings": ["lung nodule"],
            "diagnosis": "suspected lung cancer",
            "confidence": 0.85,
            "severity": "moderate",
            "reasoning": "Irregular nodule in right upper lobe",
        })
        result = parser.parse(response)
        assert result.prediction == "suspected lung cancer"
        assert result.confidence == 0.85
        assert result.metadata["parse_method"] == "json"

    def test_parse_json_in_markdown(self):
        parser = ResponseParser()
        response = '```json\n{"diagnosis": "normal", "confidence": 0.95}\n```'
        result = parser.parse(response)
        assert result.prediction == "normal"
        assert result.confidence == 0.95

    def test_parse_fallback(self):
        parser = ResponseParser()
        response = "This is a normal CT scan with no abnormalities."
        result = parser.parse(response)
        assert result.prediction != ""
        assert result.metadata["parse_method"] in ("regex", "fallback")

    def test_parse_regex_extraction(self):
        parser = ResponseParser()
        response = "Diagnosis: pneumonia\nConfidence: 0.7\nSeverity: mild"
        result = parser.parse(response)
        assert result.prediction == "pneumonia"
        assert result.confidence == 0.7
