from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from tableseq.coordinates import extract_quantized_boxes, quantized_boxes_to_original_pixels

X_TOKEN_RE = re.compile(r"^<x_\d+>$")
Y_TOKEN_RE = re.compile(r"^<y_\d+>$")


@dataclass
class BoxMetricResult:
    n_gt_boxes: int
    n_pred_boxes: int
    box_iou: Optional[float]
    box_iou_matched: Optional[float]
    box_f1_50: Optional[float]
    box_f1_75: Optional[float]
    box_l1: Optional[float]
    box_count_abs_diff: int

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {
            f"{prefix}n_gt_boxes": float(self.n_gt_boxes),
            f"{prefix}n_pred_boxes": float(self.n_pred_boxes),
            f"{prefix}box_count_abs_diff": float(self.box_count_abs_diff),
        }
        if self.box_iou is not None:
            out[f"{prefix}box_iou"] = float(self.box_iou)
        if self.box_iou_matched is not None:
            out[f"{prefix}box_iou_matched"] = float(self.box_iou_matched)
        if self.box_f1_50 is not None:
            out[f"{prefix}box_f1_50"] = float(self.box_f1_50)
        if self.box_f1_75 is not None:
            out[f"{prefix}box_f1_75"] = float(self.box_f1_75)
        if self.box_l1 is not None:
            out[f"{prefix}box_l1"] = float(self.box_l1)
        return out


def _sanitize_boxes(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    boxes = boxes.astype(np.float32).copy()
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1)


def _pairwise_iou_ordered(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n = min(len(gt), len(pred))
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    gt = _sanitize_boxes(gt[:n])
    pred = _sanitize_boxes(pred[:n])

    ix1 = np.maximum(gt[:, 0], pred[:, 0])
    iy1 = np.maximum(gt[:, 1], pred[:, 1])
    ix2 = np.minimum(gt[:, 2], pred[:, 2])
    iy2 = np.minimum(gt[:, 3], pred[:, 3])

    inter_w = np.maximum(ix2 - ix1, 0.0)
    inter_h = np.maximum(iy2 - iy1, 0.0)
    inter = inter_w * inter_h

    area_gt = np.maximum(gt[:, 2] - gt[:, 0], 0.0) * np.maximum(gt[:, 3] - gt[:, 1], 0.0)
    area_pred = np.maximum(pred[:, 2] - pred[:, 0], 0.0) * np.maximum(pred[:, 3] - pred[:, 1], 0.0)
    union = area_gt + area_pred - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def compute_box_metrics_from_strings(
    reference: str,
    prediction: str,
    image_scale_x: float = 1.0,
    image_scale_y: Optional[float] = None,
) -> BoxMetricResult:
    """Compute order-based box metrics from generated TableSeq strings.

    Missing or extra boxes are penalized because the main IoU/F1 denominator is
    ``max(n_gt, n_pred)``. ``box_iou_matched`` reports IoU only over matched
    positions and is therefore less strict.
    """
    if image_scale_y is None:
        image_scale_y = image_scale_x

    # Convert both strings to original-image pixels using the fixed five-pixel
    # model-space quantum. IoU/F1 are unchanged by this shared scaling, while
    # L1 becomes directly comparable across resolution experiments.
    gt = quantized_boxes_to_original_pixels(
        extract_quantized_boxes(reference),
        scale_x=image_scale_x,
        scale_y=image_scale_y,
    )
    pred = quantized_boxes_to_original_pixels(
        extract_quantized_boxes(prediction),
        scale_x=image_scale_x,
        scale_y=image_scale_y,
    )

    n_gt = int(len(gt))
    n_pred = int(len(pred))
    denom = max(n_gt, n_pred)
    n_match = min(n_gt, n_pred)

    if denom == 0:
        return BoxMetricResult(
            n_gt_boxes=n_gt,
            n_pred_boxes=n_pred,
            box_iou=None,
            box_iou_matched=None,
            box_f1_50=None,
            box_f1_75=None,
            box_l1=None,
            box_count_abs_diff=0,
        )

    ious = _pairwise_iou_ordered(gt, pred)
    matched_iou = float(np.mean(ious)) if len(ious) else 0.0
    padded_iou = float(np.sum(ious) / float(denom))
    f1_50 = float(np.sum(ious >= 0.50) / float(denom))
    f1_75 = float(np.sum(ious >= 0.75) / float(denom))

    if n_match > 0:
        l1 = float(np.mean(np.abs(gt[:n_match] - pred[:n_match])))
    else:
        l1 = None

    return BoxMetricResult(
        n_gt_boxes=n_gt,
        n_pred_boxes=n_pred,
        box_iou=padded_iou,
        box_iou_matched=matched_iou,
        box_f1_50=f1_50,
        box_f1_75=f1_75,
        box_l1=l1,
        box_count_abs_diff=abs(n_gt - n_pred),
    )


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted({k for d in metric_dicts for k in d})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(d[key]) for d in metric_dicts if key in d and np.isfinite(float(d[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def build_coordinate_vocab_masks(tokenizer) -> tuple[Tensor, Tensor]:
    """Return boolean vocab masks for ``<x_i>`` and ``<y_j>`` token ids."""
    vocab = tokenizer.get_vocab()
    vocab_size = max(int(idx) for idx in vocab.values()) + 1
    x_mask = torch.zeros(vocab_size, dtype=torch.bool)
    y_mask = torch.zeros(vocab_size, dtype=torch.bool)
    for token, idx in vocab.items():
        idx = int(idx)
        if X_TOKEN_RE.match(str(token)):
            x_mask[idx] = True
        elif Y_TOKEN_RE.match(str(token)):
            y_mask[idx] = True
    return x_mask, y_mask


def coordinate_token_accuracy_from_logits(
    logits: Tensor,
    labels: Tensor,
    x_vocab_mask: Tensor,
    y_vocab_mask: Tensor,
) -> dict[str, float]:
    """Compute teacher-forced accuracy restricted to coordinate-token targets."""
    pred = torch.argmax(logits.detach(), dim=-1)
    labels = labels.detach()

    device = labels.device
    vocab_size = logits.size(-1)

    if x_vocab_mask.numel() < vocab_size:
        x_pad = torch.zeros(vocab_size - x_vocab_mask.numel(), dtype=torch.bool)
        y_pad = torch.zeros(vocab_size - y_vocab_mask.numel(), dtype=torch.bool)
        x_vocab_mask = torch.cat([x_vocab_mask.cpu(), x_pad], dim=0)
        y_vocab_mask = torch.cat([y_vocab_mask.cpu(), y_pad], dim=0)
    else:
        x_vocab_mask = x_vocab_mask[:vocab_size].cpu()
        y_vocab_mask = y_vocab_mask[:vocab_size].cpu()

    x_vocab_mask = x_vocab_mask.to(device)
    y_vocab_mask = y_vocab_mask.to(device)

    safe_labels = labels.clamp(min=0, max=vocab_size - 1)
    x_targets = x_vocab_mask[safe_labels]
    y_targets = y_vocab_mask[safe_labels]
    coord_targets = x_targets | y_targets

    out: dict[str, float] = {
        "coord_token_count": float(coord_targets.sum().item()),
        "x_token_count": float(x_targets.sum().item()),
        "y_token_count": float(y_targets.sum().item()),
    }

    def _stats(mask: Tensor) -> tuple[Optional[float], int]:
        total = int(mask.sum().item())
        if total == 0:
            return None, 0
        correct = int(pred.eq(labels).logical_and(mask).sum().item())
        return float(correct / total), correct

    coord_acc, coord_correct = _stats(coord_targets)
    x_acc, x_correct = _stats(x_targets)
    y_acc, y_correct = _stats(y_targets)

    out["coord_token_correct"] = float(coord_correct)
    out["x_token_correct"] = float(x_correct)
    out["y_token_correct"] = float(y_correct)

    if coord_acc is not None:
        out["coord_token_acc"] = coord_acc
    if x_acc is not None:
        out["x_token_acc"] = x_acc
    if y_acc is not None:
        out["y_token_acc"] = y_acc

    return out
