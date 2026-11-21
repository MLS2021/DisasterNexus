"""Unified DisasterNexus model combining semantic VLM and high-resolution spatial pathways."""

from __future__ import annotations

from typing import Any, Sequence

from peft import LoraConfig, get_peft_model
import torch
from torch import Tensor, nn
from transformers import Qwen3VLForConditionalGeneration

from disasternexus.models.bitemporal_fusion import SensorConditionedBitemporalFusion
from disasternexus.models.mask_decoder import MaskedTransformerDecoder
from disasternexus.models.option_heads import TaskConditionedOptionHeads
from disasternexus.models.pixel_decoder import MultiscalePixelDecoder
from disasternexus.models.spatial_backbone import SwinSpatialBackbone


class DisasterNexusModel(nn.Module):
    """Dual-pathway model for unified nine-task disaster understanding."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vocab_size: int,
        seg_token_id: int,
    ) -> None:
        super().__init__()
        mcfg = config["model"]
        self.seg_token_id = seg_token_id

        # 1. Semantic vision-language branch (Qwen3-VL + LoRA on q_proj / v_proj).
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            mcfg["qwen_model_name_or_path"],
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        qwen.resize_token_embeddings(vocab_size)
        # Untie LM head so it can be updated independently during fine-tuning.
        old_head = qwen.get_output_embeddings()
        untied_head = nn.Linear(old_head.in_features, old_head.out_features, bias=False, dtype=old_head.weight.dtype)
        untied_head.weight.data.copy_(old_head.weight.data)
        qwen.set_output_embeddings(untied_head)

        lora_cfg = LoraConfig(
            r=int(mcfg["lora"]["r"]),
            lora_alpha=int(mcfg["lora"]["alpha"]),
            lora_dropout=float(mcfg["lora"]["dropout"]),
            target_modules=list(mcfg["lora"]["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.vlm = get_peft_model(qwen, lora_cfg)
        self.vlm.base_model.model.model.visual.requires_grad_(False)
        self.vlm.base_model.model.get_output_embeddings().requires_grad_(True)
        lm_hidden_dim = int(self.vlm.config.text_config.hidden_size)

        # 2. Independent high-resolution bitemporal spatial pathway.
        self.spatial_backbone = SwinSpatialBackbone(
            img_size=int(mcfg["spatial_image_size"]),
            freeze=bool(mcfg["spatial_backbone"]["freeze"]),
        )
        self.bitemporal_fusion = SensorConditionedBitemporalFusion(
            adapter_channels=int(mcfg["bitemporal_fusion"]["adapter_channels"]),
            modalities=mcfg["bitemporal_fusion"]["modalities"],
        )
        self.pixel_decoder = MultiscalePixelDecoder(
            conv_dim=int(mcfg["pixel_decoder"]["conv_dim"]),
            mask_dim=int(mcfg["pixel_decoder"]["mask_dim"]),
            num_heads=int(mcfg["pixel_decoder"]["num_heads"]),
            dim_feedforward=int(mcfg["pixel_decoder"]["dim_feedforward"]),
            num_encoder_layers=int(mcfg["pixel_decoder"]["num_encoder_layers"]),
        )
        self.mask_decoder = MaskedTransformerDecoder(
            hidden_dim=int(mcfg["mask_decoder"]["hidden_dim"]),
            mask_dim=int(mcfg["mask_decoder"]["mask_dim"]),
            num_queries=int(mcfg["mask_decoder"]["num_queries"]),
            num_heads=int(mcfg["mask_decoder"]["num_heads"]),
            dim_feedforward=int(mcfg["mask_decoder"]["dim_feedforward"]),
            num_decoder_layers=int(mcfg["mask_decoder"]["num_decoder_layers"]),
            lm_hidden_dim=lm_hidden_dim,
            selector_temperature=float(mcfg["mask_decoder"]["selector_temperature"]),
        )

        # 3. Task-conditioned option heads for DTR, DSR, BBR.
        self.option_heads = TaskConditionedOptionHeads(
            tasks=mcfg["option_heads"]["tasks"],
            lm_hidden_dim=lm_hidden_dim,
            head_dim=int(mcfg["option_heads"]["hidden_dim"]),
        )

    def extract_seg_hidden_states(self, input_ids: Tensor, hidden_states: Tensor) -> Tensor:
        """Locate the <seg> token representation per sample (falls back to last token)."""
        bsz, seq_len, dim = hidden_states.shape
        seg_vecs = torch.zeros((bsz, dim), device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(bsz):
            positions = (input_ids[i] == self.seg_token_id).nonzero(as_tuple=False)
            pos = int(positions[-1].item()) if positions.numel() > 0 else seq_len - 1
            seg_vecs[i] = hidden_states[i, pos]
        return seg_vecs

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        labels: Tensor | None = None,
        pre_spatial: Tensor | None = None,
        post_spatial: Tensor | None = None,
        post_modalities: Sequence[str] | None = None,
        run_spatial: bool = True,
    ) -> dict[str, Any]:
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        last_hidden = vlm_outputs.hidden_states[-1]
        outputs: dict[str, Any] = {
            "text_loss": vlm_outputs.loss,
            "logits": vlm_outputs.logits,
            "last_hidden_state": last_hidden,
            "attentions": vlm_outputs.attentions,
        }

        if run_spatial and pre_spatial is not None and post_spatial is not None:
            with torch.no_grad():
                f_pre = self.spatial_backbone(pre_spatial)
                f_post = self.spatial_backbone(post_spatial)
            mods = post_modalities or ["optical"] * pre_spatial.shape[0]
            fused = self.bitemporal_fusion(f_pre, f_post, mods)
            ms_feats, pixel_embed = self.pixel_decoder(fused)
            seg_vecs = self.extract_seg_hidden_states(input_ids, last_hidden)
            mask_out = self.mask_decoder(ms_feats, pixel_embed, seg_vecs)
            outputs.update(mask_out)

        return outputs
