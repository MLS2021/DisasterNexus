"""Benchmark evaluation CLI for DisasterNexus across QA, RefSeg, and long-form tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoProcessor
import yaml

from disasternexus.data.collator import SEG_TOKEN, format_bitemporal_messages
from disasternexus.data.dataset import DisasterM3Dataset, REFSEG_TASK
from disasternexus.metrics import compute_segmentation_metrics, exact_match
from disasternexus.models.model import DisasterNexusModel


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DisasterNexus on DisasterM3 Bench.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=["qa", "refseg", "longform", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"))
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

    dataset = DisasterM3Dataset(
        [cfg["data"]["bench_manifest"]],
        data_root=cfg["data"]["data_root"],
        semantic_size=int(cfg["model"]["semantic_image_size"]),
        spatial_size=int(cfg["model"]["spatial_image_size"]),
    )

    pred_masks, gt_masks = [], []
    qa_scores: dict[str, list[float]] = {}
    longform_preds: list[dict[str, str]] = []

    for item in dataset:
        is_refseg = item["is_refseg"]
        is_lf = item["is_longform"]

        if is_refseg and args.task in ("refseg", "all"):
            msgs = format_bitemporal_messages(item["instruction"], is_refseg=True)
            prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + f"The target region is {SEG_TOKEN}."
            inputs = processor(text=[prompt], images=[[item["pre_semantic_image"], item["post_semantic_image"]]], return_tensors="pt").to(device)
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                pre_spatial=item["pre_spatial_tensor"].unsqueeze(0).to(device),
                post_spatial=item["post_spatial_tensor"].unsqueeze(0).to(device),
                post_modalities=[item["post_modality"]],
                run_spatial=True,
            )
            best_q = int(out["pred_scores"][0].argmax().item())
            prob = F.interpolate(out["pred_masks"][0, best_q : best_q + 1].unsqueeze(0).sigmoid(), size=item["gt_mask"].shape, mode="bilinear", align_corners=False)[0, 0].cpu()
            pred_masks.append(prob)
            gt_masks.append(item["gt_mask"])

        elif not is_refseg and ((is_lf and args.task in ("longform", "all")) or (not is_lf and args.task in ("qa", "all"))):
            msgs = format_bitemporal_messages(item["instruction"], is_refseg=False)
            prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[[item["pre_semantic_image"], item["post_semantic_image"]]], return_tensors="pt").to(device)
            max_new = int(cfg["evaluation"]["max_new_tokens_longform"] if is_lf else cfg["evaluation"]["max_new_tokens_qa"])
            gen = model.vlm.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            reply = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            if is_lf:
                longform_preds.append({"sample_id": item["sample_id"], "task_type": item["task_type"], "prediction": reply, "reference": item["answer"]})
            else:
                qa_scores.setdefault(item["task_type"], []).append(exact_match(reply, item["answer"]))

    results: dict[str, object] = {}
    if pred_masks:
        results["refseg"] = compute_segmentation_metrics(pred_masks, gt_masks, threshold=float(cfg["evaluation"]["mask_threshold"]))
    if qa_scores:
        per_task = {k: 100.0 * sum(v) / len(v) for k, v in qa_scores.items()}
        results["qa"] = {"per_task_em": per_task, "macro_em": sum(per_task.values()) / len(per_task)}
    if longform_preds:
        results["longform_predictions_count"] = len(longform_preds)

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
