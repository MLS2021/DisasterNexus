"""Sensor-conditioned bitemporal adapters and residual feature fusion."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class SensorResidualAdapter(nn.Module):
    """Per-scale modality adapter with zero-initialized channel-wise residual scaling."""

    def __init__(self, in_channels: int, bottleneck_channels: int = 128) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=bottleneck_channels)
        self.act = nn.GELU()
        self.dw_conv = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            groups=bottleneck_channels,
            bias=False,
        )
        self.pw_conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, bottleneck_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        z = self.proj(x)
        res = self.pw_conv(self.dw_conv(self.act(self.norm(z))))
        return z + self.gamma * res


class BitemporalScaleFusion(nn.Module):
    """Fuse adapted pre/post features with explicit absolute difference and zero-init residual."""

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 128,
        modalities: Sequence[str] = ("optical", "sar"),
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict({
            mod: SensorResidualAdapter(in_channels, bottleneck_channels)
            for mod in modalities
        })
        self.fusion_map = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 3, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=bottleneck_channels),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        f_pre: Tensor,
        f_post: Tensor,
        post_modalities: Sequence[str],
    ) -> Tensor:
        z_pre_list: list[Tensor] = []
        z_post_list: list[Tensor] = []
        for idx, mod in enumerate(post_modalities):
            adapter = self.adapters[mod if mod in self.adapters else "optical"]
            z_pre_list.append(adapter(f_pre[idx : idx + 1]))
            z_post_list.append(adapter(f_post[idx : idx + 1]))
        z_pre = torch.cat(z_pre_list, dim=0)
        z_post = torch.cat(z_post_list, dim=0)
        concat = torch.cat([z_pre, z_post, torch.abs(z_post - z_pre)], dim=1)
        return f_post + self.out_proj(self.fusion_map(concat))


class SensorConditionedBitemporalFusion(nn.Module):
    """Apply sensor-conditioned bitemporal fusion across all four pyramid scales."""

    def __init__(
        self,
        in_channels: dict[str, int] | None = None,
        adapter_channels: int = 128,
        modalities: Sequence[str] = ("optical", "sar"),
    ) -> None:
        super().__init__()
        channels = in_channels or {"res2": 128, "res3": 256, "res4": 512, "res5": 1024}
        self.level_fusions = nn.ModuleDict({
            name: BitemporalScaleFusion(ch, adapter_channels, modalities)
            for name, ch in channels.items()
        })

    def forward(
        self,
        pre_features: dict[str, Tensor],
        post_features: dict[str, Tensor],
        post_modalities: Sequence[str],
    ) -> dict[str, Tensor]:
        return {
            name: fusion(pre_features[name], post_features[name], post_modalities)
            for name, fusion in self.level_fusions.items()
        }
