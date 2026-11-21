"""Pure-PyTorch multiscale deformable attention pixel decoder."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SinePositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding matching standard DETR/Mask2Former geometry."""

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        y_embed = torch.arange(1, height + 1, device=device, dtype=torch.float32).unsqueeze(1).expand(height, width)
        x_embed = torch.arange(1, width + 1, device=device, dtype=torch.float32).unsqueeze(0).expand(height, width)
        y_embed = y_embed / (height + 1e-6) * self.scale
        x_embed = x_embed / (width + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
        return pos.to(dtype=dtype)


class PurePyTorchMSDeformAttn(nn.Module):
    """Pure-PyTorch multi-scale deformable attention using bilinear grid sampling."""

    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> Tensor:
        bsz, len_q, _ = query.shape
        value = self.value_proj(value).view(bsz, -1, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(bsz, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attn_weights = self.attention_weights(query).view(bsz, len_q, self.n_heads, self.n_levels * self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1).view(bsz, len_q, self.n_heads, self.n_levels, self.n_points)

        normalizer = torch.tensor(
            [[w, h] for (h, w) in spatial_shapes],
            device=query.device,
            dtype=query.dtype,
        )
        sampling_locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]

        # Split flattened value tokens back into per-level spatial grids for grid_sample.
        level_sizes = [h * w for (h, w) in spatial_shapes]
        value_splits = torch.split(value, level_sizes, dim=1)
        sampled_list: list[Tensor] = []

        for lvl, (h, w) in enumerate(spatial_shapes):
            val_lvl = value_splits[lvl].permute(0, 2, 3, 1).reshape(bsz * self.n_heads, self.head_dim, h, w)
            grid_lvl = sampling_locations[:, :, :, lvl, :, :].permute(0, 2, 1, 3, 4)
            grid_lvl = grid_lvl.reshape(bsz * self.n_heads, len_q, self.n_points, 2)
            grid_lvl = 2.0 * grid_lvl - 1.0
            sampled = F.grid_sample(
                val_lvl.float(),
                grid_lvl.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).to(dtype=query.dtype)
            sampled_list.append(sampled)

        # [B*H, D_h, L_q, L_lvl, P] -> [B, L_q, H, D_h]
        stacked = torch.stack(sampled_list, dim=-2)
        stacked = stacked.view(bsz, self.n_heads, self.head_dim, len_q, self.n_levels, self.n_points)
        stacked = stacked.permute(0, 3, 1, 4, 5, 2)
        out = (stacked * attn_weights.unsqueeze(-1)).sum(dim=(3, 4)).reshape(bsz, len_q, self.d_model)
        return self.output_proj(out)


class DeformableEncoderLayer(nn.Module):
    """Single multiscale deformable transformer encoder block."""

    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, dim_ffn: int = 1024) -> None:
        super().__init__()
        self.self_attn = PurePyTorchMSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ffn, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> Tensor:
        src = self.norm1(src + self.self_attn(src + pos, reference_points, src, spatial_shapes))
        src = self.norm2(src + self.ffn(src))
        return src


class MultiscalePixelDecoder(nn.Module):
    """Transform fused bitemporal pyramid into multiscale query features and dense pixel embedding."""

    def __init__(
        self,
        in_channels: dict[str, int] | None = None,
        conv_dim: int = 256,
        mask_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 6,
    ) -> None:
        super().__init__()
        channels = in_channels or {"res2": 128, "res3": 256, "res4": 512, "res5": 1024}
        self.in_features = ("res3", "res4", "res5")
        self.conv_dim = conv_dim

        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[name], conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim),
            )
            for name in self.in_features
        ])
        self.pos_enc = SinePositionalEncoding2D(conv_dim // 2)
        self.level_embed = nn.Parameter(torch.randn(len(self.in_features), conv_dim) * 0.02)

        self.encoder_layers = nn.ModuleList([
            DeformableEncoderLayer(
                d_model=conv_dim,
                n_levels=len(self.in_features),
                n_heads=num_heads,
                dim_ffn=dim_feedforward,
            )
            for _ in range(num_encoder_layers)
        ])

        # FPN top-down fusion with res2 for high-resolution mask embedding.
        self.lateral_res2 = nn.Sequential(
            nn.Conv2d(channels["res2"], conv_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, conv_dim),
        )
        self.output_res2 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, conv_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)

    @staticmethod
    def _build_reference_points(
        spatial_shapes: Sequence[tuple[int, int]],
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        ref_list: list[Tensor] = []
        for h, w in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device, dtype=torch.float32) / h,
                torch.linspace(0.5, w - 0.5, w, device=device, dtype=torch.float32) / w,
                indexing="ij",
            )
            ref = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), dim=-1)
            ref_list.append(ref)
        refs = torch.cat(ref_list, dim=0).unsqueeze(0).expand(bsz, -1, -1)
        return refs[:, :, None, :].repeat(1, 1, len(spatial_shapes), 1).to(dtype=dtype)

    def forward(self, features: dict[str, Tensor]) -> tuple[list[Tensor], Tensor]:
        proj_feats = [proj(features[name]) for proj, name in zip(self.input_projs, self.in_features)]
        bsz = proj_feats[0].shape[0]
        spatial_shapes = [(f.shape[2], f.shape[3]) for f in proj_feats]

        src_flat: list[Tensor] = []
        pos_flat: list[Tensor] = []
        for lvl, feat in enumerate(proj_feats):
            h, w = feat.shape[2], feat.shape[3]
            pos = self.pos_enc(h, w, feat.device, feat.dtype).expand(bsz, -1, -1, -1)
            src_flat.append(feat.flatten(2).transpose(1, 2))
            pos_lvl = pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1).to(dtype=feat.dtype)
            pos_flat.append(pos_lvl)

        src = torch.cat(src_flat, dim=1)
        pos = torch.cat(pos_flat, dim=1)
        ref_pts = self._build_reference_points(spatial_shapes, bsz, src.device, src.dtype)

        for layer in self.encoder_layers:
            src = layer(src, pos, ref_pts, spatial_shapes)

        splits = torch.split(src, [h * w for h, w in spatial_shapes], dim=1)
        encoded_maps = [
            s.transpose(1, 2).reshape(bsz, self.conv_dim, h, w)
            for s, (h, w) in zip(splits, spatial_shapes)
        ]
        # Top-down fuse res3 encoded feature with res2 lateral feature.
        p2 = self.lateral_res2(features["res2"])
        p2 = p2 + F.interpolate(encoded_maps[0], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        pixel_embed = self.mask_features(self.output_res2(p2))
        # Order from coarse to fine (res5 -> res4 -> res3) for decoder layers.
        multi_scale_features = [encoded_maps[2], encoded_maps[1], encoded_maps[0]]
        return multi_scale_features, pixel_embed
