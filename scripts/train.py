"""Multi-stage training entrypoint for DisasterNexus (prefix -> joint -> continuation)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
import yaml

from disasternexus.data.collator import DisasterNexusCollator, SEG_TOKEN
from disasternexus.data.dataset import DisasterM3Dataset
from disasternexus.data.sampler import TemperatureTaskSampler
from disasternexus.losses import compute_joint_loss, compute_mask_loss
from disasternexus.models.model import DisasterNexusModel


def build_optimizer(
    model: DisasterNexusModel,
    lr_cfg: dict[str, float],
    *,
    stage: str,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Configure module-specific learning rates for the requested training stage."""
    lora_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    seg_params: list[nn.Parameter] = []
    opt_params: list[nn.Parameter] = []

    lm_head = model.vlm.base_model.model.get_output_embeddings()
    lm_head_ids = {id(p) for p in lm_head.parameters()}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in lm_head_ids:
            head_params.append(param)
        elif name.startswith("vlm."):
            lora_params.append(param)
        elif name.startswith("option_heads."):
            opt_params.append(param)
        else:
            seg_params.append(param)

    param_groups = [
        {"params": lora_params, "lr": float(lr_cfg["lora"])},
        {"params": head_params, "lr": float(lr_cfg["lm_head"])},
    ]
    if stage != "continuation":
        param_groups.extend([
            {"params": seg_params, "lr": float(lr_cfg["segmentation"])},
            {"params": opt_params, "lr": float(lr_cfg["option_heads"])},
        ])
    else:
        for p in seg_params + opt_params:
            p.requires_grad_(False)

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def save_checkpoint(model: DisasterNexusModel, output_dir: Path, tag: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{tag}.pt"
    trainable = {k: v.detach().cpu() for k, v in model.state_dict().items() if v.requires_grad or "lora_" in k or "bitemporal_fusion" in k or "pixel_decoder" in k or "mask_decoder" in k or "option_heads" in k}
    torch.save({"model": trainable}, ckpt_path)
    return ckpt_path


def run_stage(
    *,
    stage: str,
    model: DisasterNexusModel,
    dataset: DisasterM3Dataset,
    collator: DisasterNexusCollator,
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    tcfg = cfg["training"]
    micro_bs = int(tcfg["micro_batch_size"])
    grad_accum = int(tcfg["gradient_accumulation_prefix"] if stage == "prefix" else tcfg["gradient_accumulation_main"])

    if stage == "prefix":
        total_updates = int(tcfg["prefix_steps"])
    elif stage == "continuation":
        total_updates = int(tcfg["continuation_steps"])
    else:
        total_updates = math.ceil(len(dataset) * int(tcfg["main_virtual_epochs"]) / (micro_bs * grad_accum))

    num_samples = total_updates * micro_bs * grad_accum
    sampler = TemperatureTaskSampler(
        [s.task_type for s in dataset.samples],
        stage=stage,
        num_samples=num_samples,
        seed=int(cfg["data"]["seed"]),
    )
    loader = DataLoader(dataset, batch_size=micro_bs, sampler=sampler, collate_fn=collator, num_workers=4, pin_memory=True)

    optimizer = build_optimizer(model, tcfg["lr"][stage], stage=stage, weight_decay=float(tcfg["weight_decay"]))
    warmup = 50 if stage in ("prefix", "continuation") else 500
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_updates)

    lcfg = tcfg["losses"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for micro_idx, batch in enumerate(loader, start=1):
        is_refseg = batch["is_refseg"].to(device)
        gt_masks = batch["gt_masks"].to(device)

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch.get("pixel_values").to(device) if batch.get("pixel_values") is not None else None,
            image_grid_thw=batch.get("image_grid_thw").to(device) if batch.get("image_grid_thw") is not None else None,
            labels=batch["labels"].to(device),
            pre_spatial=batch["pre_spatial"].to(device),
            post_spatial=batch["post_spatial"].to(device),
            post_modalities=batch["post_modalities"],
            run_spatial=(stage != "continuation" and bool(is_refseg.any().item())),
        )

        mask_loss = None
        if stage != "continuation" and is_refseg.any():
            mask_loss = compute_mask_loss(
                outputs["pred_masks"][is_refseg],
                outputs["pred_scores"][is_refseg],
                [m[is_refseg] for m in outputs["aux_masks"]],
                [s[is_refseg] for s in outputs["aux_scores"]],
                gt_masks[is_refseg],
                bce_weight=float(lcfg["bce_weight"]),
                dice_weight=float(lcfg["dice_weight"]),
                query_ce_weight=float(lcfg["query_ce_weight"]),
            )

        loss = compute_joint_loss(
            text_loss=outputs["text_loss"],
            recognition_loss=None,
            mask_loss=mask_loss,
            attention_loss=None,
            batch_size=micro_bs,
            num_recognition=int(batch["is_recognition"].sum().item()),
            num_refseg=int(is_refseg.sum().item()),
            attention_weight=float(lcfg["attention_weight"]),
        )
        (loss / grad_accum).backward()

        if micro_idx % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0 or step == total_updates:
                print(f"[{stage}] step {step}/{total_updates} | loss={loss.item():.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DisasterNexus across prefix, main, and continuation stages.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(cfg["model"]["qwen_model_name_or_path"])
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    model = DisasterNexusModel(cfg, vocab_size=len(processor.tokenizer), seg_token_id=seg_token_id).to(device)
    dataset = DisasterM3Dataset(
        [cfg["data"]["train_manifest"], cfg["data"]["train_refseg_manifest"]],
        data_root=cfg["data"]["data_root"],
        semantic_size=int(cfg["model"]["semantic_image_size"]),
        spatial_size=int(cfg["model"]["spatial_image_size"]),
    )
    collator = DisasterNexusCollator(processor)

    out_dir = Path(cfg["training"]["output_dir"])
    for stage in ("prefix", "main", "continuation"):
        run_stage(stage=stage, model=model, dataset=dataset, collator=collator, cfg=cfg, device=device)
        save_checkpoint(model, out_dir, f"stage_{stage}")
    save_checkpoint(model, out_dir, "last")


if __name__ == "__main__":
    main()
