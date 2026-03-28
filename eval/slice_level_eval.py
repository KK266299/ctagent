# ============================================================================
# 模块职责: CQ500 逐 Slice 级别评测
#   每个 slice 独立调用 VLM API → 与 per-slice GT 对比 → 计算 slice-level 指标
#   支持按 lesion_present / bhx_coverage 分组输出指标
# ============================================================================
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eval.cq500_api_eval import (
    call_api_for_case,
    encode_slices_b64,
    CaseResult,
)
from eval.cq500_labels import DIAGNOSIS_LABELS
from eval.cq500_manifest import EvalCase, SliceEntry
from eval.metrics import compute_multilabel_metrics
from eval.slice_labels import SliceLabelEntry, SliceLabelStore
from llm.api_client import BaseLLMClient
from llm.response_parser import CQ500DiagnosisResult, parse_cq500_diagnosis

logger = logging.getLogger(__name__)


@dataclass
class SliceResult:
    """单个 slice 的评测结果。"""
    patient_id: str
    series: str
    slice_name: str
    input_type: str
    gt_labels: dict[str, int]
    pred: CQ500DiagnosisResult
    lesion_present: int = 0
    bhx_coverage: bool = True
    raw_response_text: str = ""
    api_latency_sec: float = 0.0
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


def evaluate_single_slice(
    llm_client: BaseLLMClient,
    slice_entry: SliceEntry,
    gt_labels: dict[str, int],
    input_type: str,
    mu_water: float = 0.192,
    window_center: float = 40.0,
    window_width: float = 80.0,
) -> tuple[CQ500DiagnosisResult, str, float, str | None, dict[str, int]]:
    """对单张 slice 调用 VLM API。"""
    images_b64 = encode_slices_b64(
        [slice_entry], input_type, mu_water, window_center, window_width,
    )

    t0 = time.time()
    response, _, error = call_api_for_case(llm_client, images_b64)
    latency = time.time() - t0

    if error or response is None:
        return (
            CQ500DiagnosisResult(
                predictions={lbl: 0 for lbl in DIAGNOSIS_LABELS},
                confidence={lbl: 0.0 for lbl in DIAGNOSIS_LABELS},
                parse_success=False,
            ),
            "",
            latency,
            error,
            {},
        )

    parsed = parse_cq500_diagnosis(response.text)
    return parsed, response.text, latency, None, response.usage


def run_slice_eval(
    llm_client: BaseLLMClient,
    cases: list[EvalCase],
    slice_labels: SliceLabelStore,
    input_types: list[str],
    mu_water: float = 0.192,
    window_center: float = 40.0,
    window_width: float = 80.0,
    rate_limit_sec: float = 1.0,
    bhx_only: bool = False,
) -> list[SliceResult]:
    """逐 slice 评测：每个 slice 独立调用 API。"""
    all_results: list[SliceResult] = []

    slice_count = 0
    for case in cases:
        for se in case.slices:
            slice_count += 1
    total = slice_count * len(input_types)
    done = 0

    for case in cases:
        for se in case.slices:
            slice_name = se.slice_dir.name
            label_entry = slice_labels.get_entry(
                case.patient_id, case.series, slice_name,
            )

            if label_entry is None:
                logger.debug(
                    "No slice label for %s/%s/%s, skipping",
                    case.patient_id, case.series, slice_name,
                )
                continue

            if bhx_only and not label_entry.bhx_coverage:
                continue

            gt = label_entry.labels

            for itype in input_types:
                done += 1
                if done % 20 == 0 or done == total:
                    logger.info("[%d/%d] %s/%s/%s / %s",
                                done, total, case.patient_id, case.series, slice_name, itype)

                parsed, raw_text, latency, error, usage = evaluate_single_slice(
                    llm_client, se, gt, itype,
                    mu_water, window_center, window_width,
                )

                result = SliceResult(
                    patient_id=case.patient_id,
                    series=case.series,
                    slice_name=slice_name,
                    input_type=itype,
                    gt_labels=gt,
                    pred=parsed,
                    lesion_present=label_entry.lesion_present,
                    bhx_coverage=label_entry.bhx_coverage,
                    raw_response_text=raw_text,
                    api_latency_sec=latency,
                    error=error,
                    usage=usage,
                )
                all_results.append(result)

                if rate_limit_sec > 0:
                    time.sleep(rate_limit_sec)

    return all_results


# ---------------------------------------------------------------------------
# 指标汇总
# ---------------------------------------------------------------------------

def aggregate_slice_results(
    results: list[SliceResult],
    input_types: list[str],
    split_by_lesion: bool = True,
) -> dict[str, Any]:
    """汇总 slice-level 指标，支持按 lesion/non-lesion 分组。"""
    summary: dict[str, Any] = {}

    for itype in input_types:
        type_results = [r for r in results if r.input_type == itype]
        if not type_results:
            continue

        valid = [r for r in type_results if r.error is None and r.pred.parse_success]

        itype_summary: dict[str, Any] = {
            "all": _compute_metrics_for_group(valid),
            "n_total": len(type_results),
            "n_valid": len(valid),
            "n_failed": len(type_results) - len(valid),
            "mean_latency_sec": round(
                float(np.mean([r.api_latency_sec for r in type_results])), 2
            ) if type_results else 0.0,
        }

        if split_by_lesion:
            lesion_results = [r for r in valid if r.lesion_present == 1]
            normal_results = [r for r in valid if r.lesion_present == 0]

            if lesion_results:
                itype_summary["lesion_positive"] = _compute_metrics_for_group(lesion_results)
                itype_summary["lesion_positive"]["n"] = len(lesion_results)
            if normal_results:
                itype_summary["lesion_negative"] = _compute_metrics_for_group(normal_results)
                itype_summary["lesion_negative"]["n"] = len(normal_results)

        summary[itype] = itype_summary

    return {"per_input_type": summary}


def _compute_metrics_for_group(results: list[SliceResult]) -> dict[str, Any]:
    if not results:
        return {}
    y_true = np.array([[r.gt_labels[lbl] for lbl in DIAGNOSIS_LABELS] for r in results])
    y_pred = np.array([[r.pred.predictions.get(lbl, 0) for lbl in DIAGNOSIS_LABELS] for r in results])
    y_prob = np.array([[r.pred.confidence.get(lbl, 0.5) for lbl in DIAGNOSIS_LABELS] for r in results])
    return compute_multilabel_metrics(y_true, y_pred, DIAGNOSIS_LABELS, y_prob)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def save_slice_eval_outputs(
    results: list[SliceResult],
    summary: dict[str, Any],
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "slice_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out / "slice_predictions.jsonl", "w") as f:
        for r in results:
            record = {
                "patient_id": r.patient_id,
                "series": r.series,
                "slice_name": r.slice_name,
                "input_type": r.input_type,
                "lesion_present": r.lesion_present,
                "bhx_coverage": r.bhx_coverage,
                "gt": r.gt_labels,
                "predictions": r.pred.predictions,
                "confidence": r.pred.confidence,
                "reasoning": r.pred.reasoning,
                "parse_success": r.pred.parse_success,
                "error": r.error,
                "api_latency_sec": r.api_latency_sec,
                "usage": r.usage,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    import csv as _csv
    csv_path = out / "slice_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "input_type", "group", "mean_accuracy", "macro_f1", "micro_f1",
            "mean_auroc", "n",
        ])
        for itype, m in summary.get("per_input_type", {}).items():
            for group_name in ["all", "lesion_positive", "lesion_negative"]:
                gm = m.get(group_name)
                if not gm:
                    continue
                n = gm.get("n", m.get("n_valid", 0))
                writer.writerow([
                    itype, group_name,
                    gm.get("mean_accuracy"),
                    gm.get("macro_f1"),
                    gm.get("micro_f1"),
                    gm.get("mean_auroc"),
                    n,
                ])

    logger.info("Saved slice eval outputs to %s", out)
