"""Unit tests verifying DisasterNexus spatial modules, fusion, decoders, losses, and metrics."""

from __future__ import annotations

import torch

from disasternexus.data.dataset import split_records_by_event
from disasternexus.data.sampler import TemperatureTaskSampler
from disasternexus.losses import compute_joint_loss, compute_mask_loss
from disasternexus.metrics import compute_segmentation_metrics, exact_match
from disasternexus.models.bitemporal_fusion import SensorConditionedBitemporalFusion
from disasternexus.models.mask_decoder import MaskedTransformerDecoder
from disasternexus.models.option_heads import TaskConditionedOptionHeads
from disasternexus.models.pixel_decoder import MultiscalePixelDecoder
from disasternexus.models.spatial_backbone import SwinSpatialBackbone


def test_spatial_pipeline_and_losses() -> None:
    backbone = SwinSpatialBackbone(img_size=256, pretrained=False, freeze=True)
    # Smoke test with 384x384 or 1024x1024; Swin-B window12 works cleanly at 384x384 or 1024x1024.
    backbone_384 = SwinSpatialBackbone(img_size=384, pretrained=False, freeze=True)
    pre = torch.randn(2, 3, 384, 384)
    post = torch.randn(2, 3, 384, 384)

    with torch.no_grad():
        f_pre = backbone_384(pre)
        f_post = backbone_384(post)
    assert set(f_pre.keys()) == {"res2", "res3", "res4", "res5"}

    fusion = SensorConditionedBitemporalFusion()
    fused = fusion(f_pre, f_post, ["optical", "sar"])
    assert fused["res2"].shape == f_post["res2"].shape

    pixel_dec = MultiscalePixelDecoder(num_encoder_layers=2)
    ms_feats, pixel_embed = pixel_dec(fused)
    assert len(ms_feats) == 3

    mask_dec = MaskedTransformerDecoder(num_queries=32, num_decoder_layers=4, lm_hidden_dim=128)
    seg_hidden = torch.randn(2, 128)
    mask_out = mask_dec(ms_feats, pixel_embed, seg_hidden)
    assert mask_out["pred_masks"].shape[:2] == (2, 32)
    assert mask_out["pred_scores"].shape == (2, 32)

    gt = (torch.rand(2, 96, 96) > 0.5).float()
    m_loss = compute_mask_loss(
        mask_out["pred_masks"],
        mask_out["pred_scores"],
        mask_out["aux_masks"],
        mask_out["aux_scores"],
        gt,
    )
    j_loss = compute_joint_loss(
        text_loss=torch.tensor(1.0, requires_grad=True),
        recognition_loss=None,
        mask_loss=m_loss,
        attention_loss=None,
        batch_size=2,
        num_recognition=0,
        num_refseg=2,
    )
    j_loss.backward()


def test_option_heads_sampler_and_metrics() -> None:
    heads = TaskConditionedOptionHeads(["Disaster Type Recognition"], lm_hidden_dim=64, head_dim=32)
    logits = heads("Disaster Type Recognition", torch.randn(64), torch.randn(4, 64))
    assert logits.shape == (4,)

    records = [{"event": f"ev_{i // 3}", "id": i} for i in range(12)]
    tr, va = split_records_by_event(records, val_fraction=0.25, seed=42)
    assert tr and va and set(r["event"] for r in tr).isdisjoint(r["event"] for r in va)

    sampler = TemperatureTaskSampler(["Referring Expression Segmentation"] * 5 + ["Disaster Type Recognition"] * 5, stage="main", num_samples=10)
    assert len(list(sampler)) == 10

    assert exact_match("Flood", "flood.") == 1.0
    seg_metrics = compute_segmentation_metrics([torch.ones(16, 16)], [torch.ones(16, 16)])
    assert seg_metrics["cIoU"] == 100.0


if __name__ == "__main__":
    test_spatial_pipeline_and_losses()
    test_option_heads_sampler_and_metrics()
    print("All DisasterNexus unit tests PASSED!")
