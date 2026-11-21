"""Frozen high-resolution Swin-B spatial feature pyramid."""

from __future__ import annotations

import timm
import torch
from torch import Tensor, nn


class SwinSpatialBackbone(nn.Module):
    """Extract multiscale NCHW feature maps (res2--res5) from high-resolution imagery."""

    FEATURE_NAMES = ("res2", "res3", "res4", "res5")
    OUT_CHANNELS = {"res2": 128, "res3": 256, "res4": 512, "res5": 1024}

    def __init__(
        self,
        *,
        model_name: str = "swin_base_patch4_window12_384",
        img_size: int = 1024,
        pretrained: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
        )
        if freeze:
            self.requires_grad_(False)
            self.eval()

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)
        pyramid: dict[str, Tensor] = {}
        for name, feat in zip(self.FEATURE_NAMES, features):
            # timm Swin returns NHWC tensors; convert to NCHW.
            if feat.ndim == 4 and feat.shape[-1] == self.OUT_CHANNELS[name]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            pyramid[name] = feat
        return pyramid
