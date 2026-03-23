# ============================================================================
# 模块职责: 多标签分类评测指标
#   per-label accuracy / sensitivity / specificity
#   macro-F1 / micro-F1 / AUROC
#   degraded drop / restored recovery
# ============================================================================
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-8


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, dict[str, float]]:
    """计算每个标签的 accuracy / precision / recall / F1 / specificity。

    Args:
        y_true: (N, L) 二值 GT
        y_pred: (N, L) 二值预测
        label_names: 长度 L 的标签名列表
    """
    results = {}
    for i, name in enumerate(label_names):
        tp = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
        tn = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)))
        fp = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
        fn = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))

        total = tp + tn + fp + fn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, EPSILON)
        spec = tn / max(tn + fp, 1)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "support_pos": tp + fn,
            "support_neg": tn + fp,
        }
    return results


def macro_f1(per_label: dict[str, dict[str, float]]) -> float:
    f1s = [v["f1"] for v in per_label.values()]
    return round(float(np.mean(f1s)), 4) if f1s else 0.0


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return round(2 * prec * rec / max(prec + rec, EPSILON), 4)


def auroc_per_label(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> dict[str, float | None]:
    """计算每标签 AUROC（需要 confidence 输出）。"""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("sklearn not available, skipping AUROC")
        return {name: None for name in label_names}

    results = {}
    for i, name in enumerate(label_names):
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]
        if len(np.unique(col_true)) < 2:
            results[name] = None
        else:
            try:
                results[name] = round(float(roc_auc_score(col_true, col_prob)), 4)
            except Exception:
                results[name] = None
    return results


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """汇总所有多标签分类指标。"""
    pl = per_label_metrics(y_true, y_pred, label_names)
    result: dict[str, Any] = {
        "per_label": pl,
        "macro_f1": macro_f1(pl),
        "micro_f1": micro_f1(y_true, y_pred),
        "mean_accuracy": round(float(np.mean([v["accuracy"] for v in pl.values()])), 4),
    }
    if y_prob is not None:
        result["auroc"] = auroc_per_label(y_true, y_prob, label_names)
        valid_aucs = [v for v in result["auroc"].values() if v is not None]
        result["mean_auroc"] = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None

    return result


def compute_drop_recovery(
    clean_metrics: dict[str, Any],
    degraded_metrics: dict[str, Any],
    restored_metrics: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """计算退化 drop 和修复 recovery。"""
    c_acc = clean_metrics["mean_accuracy"]
    d_acc = degraded_metrics["mean_accuracy"]

    drop = round((c_acc - d_acc) / max(c_acc, EPSILON) * 100, 2)
    result: dict[str, float | None] = {
        "clean_accuracy": c_acc,
        "degraded_accuracy": d_acc,
        "degraded_drop_pct": drop,
    }

    if restored_metrics is not None:
        r_acc = restored_metrics["mean_accuracy"]
        gap = c_acc - d_acc
        recovery = round((r_acc - d_acc) / max(abs(gap), EPSILON) * 100, 2) if abs(gap) > EPSILON else 0.0
        result["restored_accuracy"] = r_acc
        result["restored_recovery_pct"] = recovery
    else:
        result["restored_accuracy"] = None
        result["restored_recovery_pct"] = None

    return result
