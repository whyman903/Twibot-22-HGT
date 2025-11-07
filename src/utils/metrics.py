"""Metric helpers for evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report, f1_score


def compute_auprc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    labels_np = labels.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    if probs_np.ndim == 2 and probs_np.shape[1] > 1:
        scores = probs_np[:, 1]
    else:
        scores = probs_np
    try:
        return float(average_precision_score(labels_np, scores))
    except ValueError:
        return float("nan")


def compute_metrics(labels: torch.Tensor, preds: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    metrics: Dict[str, float] = {}
    try:
        metrics["f1"] = float(f1_score(labels_np, preds_np))
    except ValueError:
        metrics["f1"] = float("nan")
    try:
        metrics["auprc"] = compute_auprc(labels, probs)
    except ValueError:
        metrics["auprc"] = float("nan")
    report = classification_report(labels_np, preds_np, output_dict=True, zero_division=0)
    for cls in ["0", "1"]:
        if cls in report:
            metrics[f"precision_{cls}"] = float(report[cls]["precision"])
            metrics[f"recall_{cls}"] = float(report[cls]["recall"])
    metrics["precision_macro"] = float(report["macro avg"]["precision"])
    metrics["recall_macro"] = float(report["macro avg"]["recall"])
    metrics["accuracy"] = float(report["accuracy"])
    return metrics
