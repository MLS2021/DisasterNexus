"""Unified inference CLI for DisasterNexus (QA, referring segmentation, and long-form)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import AutoProcessor
import yaml

from disasternexus.data.collator import SEG_TOKEN, format_bitemporal_messages
from disasternexus.data.dataset import image_to_normalized_tensor, load_rgb_image
from disasternexus.models.model import DisasterNexusModel


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run DisasterNexus inference on a pre/post image pair.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pre-image", type=Path, required=True)
    parser.add_argument("--post-image", type=Path, required=True)
    parser.add_argument("--modality", choices=["optical", "sar"], default="optical")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--task", choices=["qa", "refseg", "longform"], default="refseg")
    parser.add_argument("--mask-out", type=Path, default=Path("pred_mask.png"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(cfg["model"]["qwen_model_name_or_path"])
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)

    model = DisasterNexusModel(cfg, vocab_size=len(processor.tokenizer), seg_token_id=seg_token_id).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    sem_size = int(cfg["model"]["semantic_image_size"])
    spa_size = int(cfg["model"]["spatial_image_size"])
    pre_sem = load_rgb_image(args.pre_image, sem_size)
    post_sem = load_rgb_image(args.post_image, sem_size)

    messages = format_bitemporal_messages(args.prompt, answer=None, is_refseg=(args.task == "refseg"))
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.task == "refseg":
        prompt_text = f"{prompt_text}The target region is {SEG_TOKEN}."

    inputs = processor(text=[prompt_text], images=[[pre_sem, post_sem]], return_tensors="pt").to(device)

    if args.task == "refseg":
        pre_spa = image_to_normalized_tensor(load_rgb_image(args.pre_image, spa_size)).unsqueeze(0).to(device)
        post_spa = image_to_normalized_tensor(load_rgb_image(args.post_image, spa_size)).unsqueeze(0).to(device)
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pre_spatial=pre_spa,
            post_spatial=post_spa,
            post_modalities=[args.modality],
            run_spatial=True,
        )
        best_idx = int(out["pred_scores"][0].argmax().item())
        mask_logits = out["pred_masks"][0, best_idx : best_idx + 1].unsqueeze(0)
        probs = F.interpolate(mask_logits.sigmoid(), size=(spa_size, spa_size), mode="bilinear", align_corners=False)[0, 0]
        binary = (probs > float(cfg["evaluation"]["mask_threshold"])).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(binary).save(args.mask_out)
        print(f"Saved predicted mask to {args.mask_out}")
    else:
        max_new = int(cfg["evaluation"]["max_new_tokens_longform"] if args.task == "longform" else cfg["evaluation"]["max_new_tokens_qa"])
        generated = model.vlm.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        reply = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(reply.strip())


if __name__ == "__main__":
    main()
