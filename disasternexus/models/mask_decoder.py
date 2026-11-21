"""Masked transformer decoder with 32 candidate queries and <seg> relevance scoring."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from disasternexus.models.pixel_decoder import SinePositionalEncoding2D


class MaskedDecoderLayer(nn.Module):
    """Single masked cross-attention + self-attention + FFN block."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dim_ffn: int = 2048) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ffn, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        q = queries + query_pos
        k = memory + memory_pos
        # Prevent full-row masking NaNs by unmasking rows that are completely blocked.
        if attn_mask is not None:
            all_masked = attn_mask.all(dim=-1, keepdim=True)
            attn_mask = torch.where(all_masked, torch.zeros_like(attn_mask), attn_mask)
            attn_mask = attn_mask.repeat_interleave(self.cross_attn.num_heads, dim=0)

        cross_out, _ = self.cross_attn(query=q, key=k, value=memory, attn_mask=attn_mask)
        queries = self.norm1(queries + cross_out)

        q2 = queries + query_pos
        self_out, _ = self.self_attn(query=q2, key=q2, value=queries)
        queries = self.norm2(queries + self_out)
        queries = self.norm3(queries + self.ffn(queries))
        return queries


class MaskedTransformerDecoder(nn.Module):
    """Generate 32 candidate masks and score their relevance against the <seg> embedding."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_queries: int = 32,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_decoder_layers: int = 9,
        lm_hidden_dim: int = 2560,
        selector_temperature: float = 16.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_decoder_layers
        self.selector_temperature = selector_temperature

        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.pos_enc = SinePositionalEncoding2D(hidden_dim // 2)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([
            MaskedDecoderLayer(hidden_dim, num_heads, dim_feedforward)
            for _ in range(num_decoder_layers)
        ])

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mask_dim),
        )
        # Project language <seg> hidden state and candidate queries into a shared scoring space.
        self.seg_proj = nn.Sequential(
            nn.Linear(lm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_score_proj = nn.Linear(hidden_dim, hidden_dim)

    def _predict_masks_and_attn(
        self,
        queries: Tensor,
        pixel_embed: Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        norm_q = self.decoder_norm(queries)
        mask_embeds = self.mask_embed_head(norm_q)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeds, pixel_embed)
        attn_resized = F.interpolate(pred_masks, size=target_hw, mode="bilinear", align_corners=False)
        attn_mask = (attn_resized.sigmoid().flatten(2) < 0.5).bool()
        return pred_masks, attn_mask

    def forward(
        self,
        multi_scale_features: list[Tensor],
        pixel_embed: Tensor,
        seg_hidden_states: Tensor,
    ) -> dict[str, Tensor | list[Tensor]]:
        bsz = pixel_embed.shape[0]
        device = pixel_embed.device
        dtype = pixel_embed.dtype

        queries = self.query_feat.weight.unsqueeze(0).expand(bsz, -1, -1).to(dtype=dtype)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(bsz, -1, -1).to(dtype=dtype)

        # Prepare 3 memory scales (coarse -> fine) for cyclic layer attention.
        memories: list[Tensor] = []
        mem_poses: list[Tensor] = []
        mem_hw: list[tuple[int, int]] = []
        for lvl, feat in enumerate(multi_scale_features[:3]):
            h, w = feat.shape[2], feat.shape[3]
            pos = self.pos_enc(h, w, device, dtype).expand(bsz, -1, -1, -1)
            mem = feat.flatten(2).transpose(1, 2)
            pos_flat = pos.flatten(2).transpose(1, 2) + self.level_embed.weight[lvl].view(1, 1, -1).to(dtype=dtype)
            memories.append(mem)
            mem_poses.append(pos_flat)
            mem_hw.append((h, w))

        seg_emb = F.normalize(self.seg_proj(seg_hidden_states.to(dtype=dtype)), dim=-1)

        all_masks: list[Tensor] = []
        all_scores: list[Tensor] = []

        pred_masks, attn_mask = self._predict_masks_and_attn(queries, pixel_embed, mem_hw[0])
        for layer_idx, layer in enumerate(self.layers):
            lvl = layer_idx % len(memories)
            queries = layer(queries, query_pos, memories[lvl], mem_poses[lvl], attn_mask)
            next_lvl = (layer_idx + 1) % len(memories)
            pred_masks, attn_mask = self._predict_masks_and_attn(queries, pixel_embed, mem_hw[next_lvl])
            q_proj = F.normalize(self.query_score_proj(self.decoder_norm(queries)), dim=-1)
            scores = torch.einsum("bqc,bc->bq", q_proj, seg_emb) * math.sqrt(self.hidden_dim) / self.selector_temperature
            all_masks.append(pred_masks)
            all_scores.append(scores)

        # Final output + last 3 auxiliary outputs used for deep supervision.
        return {
            "pred_masks": all_masks[-1],
            "pred_scores": all_scores[-1],
            "aux_masks": all_masks[-4:-1],
            "aux_scores": all_scores[-4:-1],
        }
