"""Evaluation metrics for short-answer QA and referring segmentation."""

from __future__ import annotations

import re
from typing import Sequence

import torch
from torch import Tensor


def normalize_answer(text: str) -> str:
    """Normalize prediction and reference strings for exact-match evaluation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".")


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def compute_segmentation_metrics(
    pred_masks: Sequence[Tensor],
    gt_masks: Sequence[Tensor],
    *,
    threshold: float = 0.3,
) -> dict[str, float]:
    """Compute mIoU, cIoU, non-empty mIoU, and empty-target accuracy."""
    total_inter = 0.0
    total_union = 0.0
    sample_ious: list[float] = []
    nonempty_ious: list[float] = []
    empty_correct = 0
    empty_total = 0

    for pred, gt in zip(pred_masks, gt_masks):
        p_bin = (pred > threshold).bool()
        g_bin = (gt > 0.5).bool()
        inter = float((p_bin & g_bin).sum().item())
        union = float((p_bin | g_bin).sum().item())

        total_inter += inter
        total_union += union

        gt_empty = not bool(g_bin.any().item())
        pred_empty = not bool(p_bin.any().item())

        if gt_empty:
            empty_total += 1
            if pred_empty:
                empty_correct += 1
                iou = 1.0
            else:
                iou = 0.0
        else:
            iou = inter / union if union > 0 else 0.0
            nonempty_ious.append(iou)
        sample_ious.append(iou)

    n = max(len(sample_ious), 1)
    return {
        "mIoU": 100.0 * sum(sample_ious) / n,
        "cIoU": 100.0 * total_inter / total_union if total_union > 0 else 0.0,
        "nonempty_mIoU": 100.0 * sum(nonempty_ious) / max(len(nonempty_ious), 1),
        "empty_accuracy": 100.0 * empty_correct / max(empty_total, 1),
    }
