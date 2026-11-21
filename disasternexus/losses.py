"""Joint multi-task losses: text, recognition option-head, mask, and spatial attention."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def dice_loss_from_logits(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute binary Dice loss over spatial dimensions [B, H, W]."""
    probs = torch.sigmoid(logits)
    probs_flat = probs.flatten(1)
    targets_flat = targets.flatten(1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def compute_mask_loss(
    pred_masks: Tensor,
    pred_scores: Tensor,
    aux_masks: list[Tensor],
    aux_scores: list[Tensor],
    gt_masks: Tensor,
    *,
    bce_weight: float = 5.0,
    dice_weight: float = 5.0,
    query_ce_weight: float = 1.0,
) -> Tensor:
    """Select best-matching candidate query per layer and sum BCE, Dice, and query CE losses."""
    if pred_masks.shape[0] == 0:
        return pred_masks.sum() * 0.0

    total_loss = pred_masks.sum() * 0.0
    layer_outputs = [(pred_masks, pred_scores)] + list(zip(aux_masks, aux_scores))

    for masks, scores in layer_outputs:
        gt_resized = F.interpolate(gt_masks.unsqueeze(1), size=masks.shape[-2:], mode="nearest").squeeze(1)
        bsz, num_q, h, w = masks.shape

        # Match each sample's ground-truth mask to the best candidate query.
        gt_exp = gt_resized.unsqueeze(1).expand(-1, num_q, -1, -1)
        bce_all = F.binary_cross_entropy_with_logits(masks.float(), gt_exp.float(), reduction="none").mean(dim=(-2, -1))
        probs = torch.sigmoid(masks.float()).flatten(2)
        gt_flat = gt_exp.float().flatten(2)
        inter = (probs * gt_flat).sum(dim=-1)
        dice_all = 1.0 - (2.0 * inter + 1e-6) / (probs.sum(dim=-1) + gt_flat.sum(dim=-1) + 1e-6)
        cost = bce_weight * bce_all + dice_weight * dice_all
        best_q = cost.argmin(dim=1)

        batch_idx = torch.arange(bsz, device=masks.device)
        selected_masks = masks[batch_idx, best_q]
        bce_loss = F.binary_cross_entropy_with_logits(selected_masks.float(), gt_resized.float())
        dice_loss = dice_loss_from_logits(selected_masks.float(), gt_resized.float())
        ce_loss = F.cross_entropy(scores.float(), best_q)

        total_loss = total_loss + bce_weight * bce_loss + dice_weight * dice_loss + query_ce_weight * ce_loss

    return total_loss


def compute_joint_loss(
    *,
    text_loss: Tensor,
    recognition_loss: Tensor | None,
    mask_loss: Tensor | None,
    attention_loss: Tensor | None,
    batch_size: int,
    num_recognition: int,
    num_refseg: int,
    attention_weight: float = 0.01,
) -> Tensor:
    """Combine text, subset-weighted recognition, and subset-weighted segmentation losses."""
    total = text_loss
    if recognition_loss is not None and num_recognition > 0:
        total = total + (num_recognition / batch_size) * recognition_loss
    if mask_loss is not None and num_refseg > 0:
        seg_term = mask_loss
        if attention_loss is not None:
            seg_term = seg_term + attention_weight * attention_loss
        total = total + (num_refseg / batch_size) * seg_term
    return total
