# ============================================================================
# 模块职责: CQ500 闭源 API 诊断分类批量评测
#   对 clean / degraded / restored 三路数据调用 VLM API → 解析诊断 → 计算指标
#   所有 API 调用通过 llm/ 层统一封装
# ============================================================================
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from eval.cq500_labels import CQ500Labels, DIAGNOSIS_LABELS, normalize_case_id
from eval.cq500_manifest import EvalCase, SliceEntry
from eval.metrics import compute_multilabel_metrics, compute_drop_recovery
from llm.api_client import BaseLLMClient, LLMResponse
from llm.prompt_builder import CQ500_DIAGNOSIS_SYSTEM_PROMPT, build_cq500_user_prompt
from llm.response_parser import CQ500DiagnosisResult, parse_cq500_diagnosis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 图像加载与编码
# ---------------------------------------------------------------------------

def load_h5_image(h5_path: Path, dataset_key: str = "image") -> np.ndarray:
    """从 HDF5 文件加载图像数据。"""
    with h5py.File(str(h5_path), "r") as f:
        return f[dataset_key][:]


def mu_to_windowed_uint8(
    mu_image: np.ndarray,
    mu_water: float = 0.192,
    window_center: float = 40.0,
    window_width: float = 80.0,
) -> np.ndarray:
    """线衰减系数 → HU → 窗位 → uint8。"""
    hu = (mu_image / mu_water - 1.0) * 1000.0
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    normalized = (hu - lo) / (hi - lo)
    return (np.clip(normalized, 0, 1) * 255).astype(np.uint8)


_restore_fn = None


def set_restore_fn(fn: Any) -> None:
    """Set the restoration function for 'restored' mode.

    fn(mu_image: np.ndarray) -> np.ndarray
    """
    global _restore_fn
    _restore_fn = fn


def encode_slices_b64(
    slice_entries: list[SliceEntry],
    mode: str,
    mu_water: float = 0.192,
    window_center: float = 40.0,
    window_width: float = 80.0,
) -> list[str]:
    """将多个 slice 编码为 base64 PNG 列表。

    Args:
        mode: "clean" → gt.h5["image"],
              "degraded" → {idx}.h5["ma_CT"],
              "restored" → degraded + restoration pipeline
    """
    import base64
    import io
    from PIL import Image

    b64_list = []
    for se in slice_entries:
        if mode == "clean":
            mu = load_h5_image(se.gt_h5, "image")
        elif mode == "degraded":
            mu = load_h5_image(se.degraded_h5, "ma_CT")
        elif mode == "restored":
            mu = load_h5_image(se.degraded_h5, "ma_CT")
            if _restore_fn is not None:
                mu = _restore_fn(mu)
            else:
                logger.warning("No restore_fn set, falling back to clip only")
                mu = np.clip(mu, 0.0, 0.6)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        img_uint8 = mu_to_windowed_uint8(mu, mu_water, window_center, window_width)
        pil_img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return b64_list


# ---------------------------------------------------------------------------
# 单 case API 调用
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """单个 case 的评测结果。"""
    case_id: str
    input_type: str
    gt_labels: dict[str, int]
    pred: CQ500DiagnosisResult
    raw_prompt_text: str = ""
    raw_response_text: str = ""
    api_latency_sec: float = 0.0
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


def call_api_for_case(
    llm_client: BaseLLMClient,
    images_b64: list[str],
    system_prompt: str = CQ500_DIAGNOSIS_SYSTEM_PROMPT,
) -> tuple[LLMResponse | None, str, str | None]:
    """调用 VLM API 进行诊断分类。

    Returns:
        (LLMResponse, user_prompt_text, error_string_or_None)
    """
    user_text = build_cq500_user_prompt(len(images_b64))

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict[str, Any]] = []
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": content})

    try:
        response = llm_client.chat(messages)
        return response, user_text, None
    except Exception as e:
        return None, user_text, str(e)


def evaluate_single_case(
    llm_client: BaseLLMClient,
    eval_case: EvalCase,
    gt_labels: dict[str, int],
    input_type: str,
    mu_water: float = 0.192,
    window_center: float = 40.0,
    window_width: float = 80.0,
) -> CaseResult:
    """对单个 case 的一路输入执行 API 评测。"""
    images_b64 = encode_slices_b64(
        eval_case.slices, input_type, mu_water, window_center, window_width,
    )

    t0 = time.time()
    response, prompt_text, error = call_api_for_case(llm_client, images_b64)
    latency = time.time() - t0

    if error or response is None:
        logger.warning("[%s/%s] API error: %s", eval_case.case_id, input_type, error)
        return CaseResult(
            case_id=eval_case.case_id,
            input_type=input_type,
            gt_labels=gt_labels,
            pred=CQ500DiagnosisResult(
                predictions={lbl: 0 for lbl in DIAGNOSIS_LABELS},
                confidence={lbl: 0.0 for lbl in DIAGNOSIS_LABELS},
                parse_success=False,
            ),
            raw_prompt_text=prompt_text,
            raw_response_text="",
            api_latency_sec=latency,
            error=error,
        )

    parsed = parse_cq500_diagnosis(response.text)

    return CaseResult(
        case_id=eval_case.case_id,
        input_type=input_type,
        gt_labels=gt_labels,
        pred=parsed,
        raw_prompt_text=prompt_text,
        raw_response_text=response.text,
        api_latency_sec=latency,
        error=None,
        usage=response.usage,
    )


# ---------------------------------------------------------------------------
# 批量评测
# ---------------------------------------------------------------------------

def run_batch_eval(
    llm_client: BaseLLMClient,
    cases: list[EvalCase],
    labels: CQ500Labels,
    input_types: list[str],
    mu_water: float = 0.192,
    window_center: float = 40.0,
    window_width: float = 80.0,
    rate_limit_sec: float = 1.0,
) -> list[CaseResult]:
    """批量评测所有 case × 所有 input_type。"""
    all_results: list[CaseResult] = []
    total = len(cases) * len(input_types)
    done = 0

    for case in cases:
        gt = labels.get_gt(case.case_id)
        if gt is None:
            logger.warning("No GT labels for %s, skipping", case.case_id)
            continue

        for itype in input_types:
            done += 1
            logger.info("[%d/%d] %s / %s", done, total, case.case_id, itype)

            result = evaluate_single_case(
                llm_client, case, gt, itype, mu_water, window_center, window_width,
            )
            all_results.append(result)

            if rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

    return all_results


# ---------------------------------------------------------------------------
# 结果汇总与输出
# ---------------------------------------------------------------------------

def aggregate_results(
    results: list[CaseResult],
    input_types: list[str],
) -> dict[str, Any]:
    """按 input_type 分组计算指标并汇总。"""
    type_metrics: dict[str, Any] = {}

    for itype in input_types:
        type_results = [r for r in results if r.input_type == itype]
        if not type_results:
            continue

        valid = [r for r in type_results if r.error is None and r.pred.parse_success]

        y_true = np.array([[r.gt_labels[lbl] for lbl in DIAGNOSIS_LABELS] for r in valid])
        y_pred = np.array([[r.pred.predictions.get(lbl, 0) for lbl in DIAGNOSIS_LABELS] for r in valid])
        y_prob = np.array([[r.pred.confidence.get(lbl, 0.5) for lbl in DIAGNOSIS_LABELS] for r in valid])

        metrics = compute_multilabel_metrics(y_true, y_pred, DIAGNOSIS_LABELS, y_prob)
        metrics["n_total"] = len(type_results)
        metrics["n_valid"] = len(valid)
        metrics["n_failed"] = len(type_results) - len(valid)
        metrics["mean_latency_sec"] = round(
            float(np.mean([r.api_latency_sec for r in type_results])), 2
        )
        type_metrics[itype] = metrics

    summary: dict[str, Any] = {"per_input_type": type_metrics}

    if "clean" in type_metrics and "degraded" in type_metrics:
        dr = compute_drop_recovery(
            type_metrics["clean"],
            type_metrics["degraded"],
            type_metrics.get("restored"),
        )
        summary["drop_recovery"] = dr

    return summary


def select_case_studies(results: list[CaseResult], top_k: int = 10) -> list[dict[str, Any]]:
    """自动挑选典型 case study。"""
    studies: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, CaseResult]] = {}
    for r in results:
        by_case.setdefault(r.case_id, {})[r.input_type] = r

    for case_id, type_map in by_case.items():
        clean_r = type_map.get("clean")
        deg_r = type_map.get("degraded")
        if not clean_r or not deg_r:
            continue

        clean_correct = sum(
            1 for lbl in DIAGNOSIS_LABELS
            if clean_r.pred.predictions.get(lbl, 0) == clean_r.gt_labels[lbl]
        )
        deg_correct = sum(
            1 for lbl in DIAGNOSIS_LABELS
            if deg_r.pred.predictions.get(lbl, 0) == deg_r.gt_labels[lbl]
        )

        if clean_correct > deg_correct:
            studies.append({
                "case_id": case_id,
                "type": "clean_correct_degraded_wrong",
                "clean_score": clean_correct,
                "degraded_score": deg_correct,
                "diff": clean_correct - deg_correct,
            })

        conf_diffs = []
        for lbl in DIAGNOSIS_LABELS:
            c_conf = clean_r.pred.confidence.get(lbl, 0.5)
            d_conf = deg_r.pred.confidence.get(lbl, 0.5)
            conf_diffs.append(abs(c_conf - d_conf))
        max_conf_diff = max(conf_diffs) if conf_diffs else 0.0

        if max_conf_diff > 0.3:
            studies.append({
                "case_id": case_id,
                "type": "high_confidence_shift",
                "max_confidence_diff": round(max_conf_diff, 3),
            })

        rst_r = type_map.get("restored")
        if rst_r and deg_correct < clean_correct:
            rst_correct = sum(
                1 for lbl in DIAGNOSIS_LABELS
                if rst_r.pred.predictions.get(lbl, 0) == rst_r.gt_labels[lbl]
            )
            if rst_correct > deg_correct:
                studies.append({
                    "case_id": case_id,
                    "type": "restored_recovery",
                    "degraded_score": deg_correct,
                    "restored_score": rst_correct,
                })

    studies.sort(key=lambda x: x.get("diff", x.get("max_confidence_diff", 0)), reverse=True)
    return studies[:top_k]


def save_outputs(
    results: list[CaseResult],
    summary: dict[str, Any],
    case_studies: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """保存全部评测输出文件。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    import csv
    csv_path = out / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_type", "mean_accuracy", "macro_f1", "micro_f1",
                          "mean_auroc", "n_valid", "n_failed", "mean_latency_sec"])
        for itype, m in summary.get("per_input_type", {}).items():
            writer.writerow([
                itype, m.get("mean_accuracy"), m.get("macro_f1"), m.get("micro_f1"),
                m.get("mean_auroc"), m.get("n_valid"), m.get("n_failed"),
                m.get("mean_latency_sec"),
            ])

    with open(out / "predictions.jsonl", "w") as f:
        for r in results:
            record = {
                "case_id": r.case_id,
                "input_type": r.input_type,
                "gt": r.gt_labels,
                "predictions": r.pred.predictions,
                "confidence": r.pred.confidence,
                "reasoning": r.pred.reasoning,
                "parse_success": r.pred.parse_success,
                "error": r.error,
                "api_latency_sec": r.api_latency_sec,
                "usage": r.usage,
                "raw_prompt": r.raw_prompt_text,
                "raw_response": r.raw_response_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(out / "case_studies.json", "w") as f:
        json.dump(case_studies, f, indent=2, ensure_ascii=False)

    logger.info("Saved outputs to %s", out)
