"""Dataset loading, event-based splitting, and dual-resolution image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RECOGNITION_TASKS = frozenset({
    "Disaster Type Recognition",
    "Disaster Scene Recognition",
    "Disaster Bearing Bodies Recognition",
})
REFSEG_TASK = "Referring Expression Segmentation"
LONGFORM_TASKS = frozenset({
    "Disaster Report",
    "Disaster Restoration Advice",
    "disaster caption",
    "disaster restoration advice",
})


@dataclass(frozen=True)
class DisasterSample:
    """Normalized record for any of the nine DisasterM3 tasks."""

    sample_id: str
    task_type: str
    pre_image_path: str
    post_image_path: str
    post_modality: str
    instruction: str
    answer: str
    mask_path: str | None = None
    options: list[str] = field(default_factory=list)
    event: str = "unknown"


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL lines from a manifest file."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_records_by_event(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train and validation sets by disaster event without leakage."""
    by_event: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        event = str(rec.get("event") or "unknown")
        by_event.setdefault(event, []).append(rec)

    events = sorted(by_event)
    rng = random.Random(seed)
    rng.shuffle(events)

    target_val = int(round(len(records) * val_fraction))
    val_events: set[str] = set()
    val_count = 0
    for ev in events:
        if val_count >= target_val and val_events:
            break
        val_events.add(ev)
        val_count += len(by_event[ev])

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for rec in records:
        ev = str(rec.get("event") or "unknown")
        if ev in val_events:
            val_records.append(rec)
        else:
            train_records.append(rec)
    return train_records, val_records


def _resolve_path(data_root: Path, rel_or_abs: str | None) -> Path | None:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    return p if p.is_absolute() else data_root / p


def load_rgb_image(path: Path, size: int) -> Image.Image:
    """Load an image as RGB and resize to a square target resolution."""
    with Image.open(path) as img:
        return img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def image_to_normalized_tensor(image: Image.Image) -> Tensor:
    """Convert an RGB PIL image to a normalized [3, H, W] float32 tensor."""
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_binary_mask(path: Path | None, size: int) -> tuple[Tensor, bool]:
    """Load a binary target mask [H, W] and return whether the target is non-empty."""
    if path is None or not path.exists():
        return torch.zeros((size, size), dtype=torch.float32), False
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((size, size), Image.Resampling.NEAREST))
    mask = torch.from_numpy((arr > 0).astype(np.float32))
    has_target = bool(mask.sum().item() > 0)
    return mask, has_target


class DisasterM3Dataset(Dataset):
    """Multi-task bitemporal dataset serving semantic (512) and spatial (1024) inputs."""

    def __init__(
        self,
        manifest_paths: Sequence[str | Path],
        data_root: str | Path,
        *,
        semantic_size: int = 512,
        spatial_size: int = 1024,
        mask_size: int = 256,
    ) -> None:
        self.data_root = Path(data_root)
        self.semantic_size = semantic_size
        self.spatial_size = spatial_size
        self.mask_size = mask_size

        self.samples: list[DisasterSample] = []
        for manifest in manifest_paths:
            for idx, raw in enumerate(load_jsonl_records(manifest)):
                self.samples.append(
                    DisasterSample(
                        sample_id=str(raw.get("sample_id", f"{Path(manifest).stem}_{idx}")),
                        task_type=str(raw["task_type"]),
                        pre_image_path=str(raw.get("pre_image_path") or raw.get("pre_image")),
                        post_image_path=str(raw.get("post_image_path") or raw.get("post_image")),
                        post_modality=str(raw.get("post_modality", "optical")).lower(),
                        instruction=str(raw.get("instruction") or raw.get("question", "")),
                        answer=str(raw.get("answer") or raw.get("ground_truth", "")),
                        mask_path=raw.get("mask_path"),
                        options=list(raw.get("options", [])),
                        event=str(raw.get("event", "unknown")),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_path = _resolve_path(self.data_root, sample.pre_image_path)
        post_path = _resolve_path(self.data_root, sample.post_image_path)
        mask_path = _resolve_path(self.data_root, sample.mask_path)

        pre_sem = load_rgb_image(pre_path, self.semantic_size)
        post_sem = load_rgb_image(post_path, self.semantic_size)

        pre_spa = image_to_normalized_tensor(load_rgb_image(pre_path, self.spatial_size))
        post_spa = image_to_normalized_tensor(load_rgb_image(post_path, self.spatial_size))

        is_refseg = sample.task_type == REFSEG_TASK
        if is_refseg:
            gt_mask, is_nonempty = load_binary_mask(mask_path, self.mask_size)
        else:
            gt_mask = torch.zeros((self.mask_size, self.mask_size), dtype=torch.float32)
            is_nonempty = False

        return {
            "sample_id": sample.sample_id,
            "task_type": sample.task_type,
            "post_modality": sample.post_modality,
            "instruction": sample.instruction,
            "answer": sample.answer,
            "options": sample.options,
            "pre_semantic_image": pre_sem,
            "post_semantic_image": post_sem,
            "pre_spatial_tensor": pre_spa,
            "post_spatial_tensor": post_spa,
            "gt_mask": gt_mask,
            "is_refseg": is_refseg,
            "is_nonempty": is_nonempty,
            "is_recognition": sample.task_type in RECOGNITION_TASKS,
            "is_longform": sample.task_type in LONGFORM_TASKS,
        }
