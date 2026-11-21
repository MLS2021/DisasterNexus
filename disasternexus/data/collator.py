"""Mixed-task prompt construction and batch collation for Qwen3-VL + spatial inputs."""

from __future__ import annotations

from typing import Any

import torch

SEG_TOKEN = "<seg>"


def format_bitemporal_messages(
    instruction: str,
    *,
    answer: str | None = None,
    is_refseg: bool = False,
) -> list[dict[str, Any]]:
    """Build a Qwen3-VL chat conversation for pre/post imagery and instruction."""
    user_text = (
        "Pre-event image and post-event image are provided.\n"
        f"{instruction}"
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    if answer is not None:
        target_text = f"The target region is {SEG_TOKEN}." if is_refseg else answer
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        })
    return messages


class DisasterNexusCollator:
    """Collate mixed-task samples into semantic VLM inputs and spatial tensors."""

    def __init__(self, processor: Any, *, seg_token: str = SEG_TOKEN) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.seg_token = seg_token
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids(seg_token)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        image_pairs: list[list[Any]] = []

        for item in batch:
            messages = format_bitemporal_messages(
                item["instruction"],
                answer=item["answer"],
                is_refseg=item["is_refseg"],
            )
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(prompt_text)
            image_pairs.append([item["pre_semantic_image"], item["post_semantic_image"]])

        model_inputs = self.processor(
            text=texts,
            images=image_pairs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Mask user-prompt tokens so text loss applies only to assistant tokens.
        assistant_header = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        for row_idx in range(input_ids.size(0)):
            seq = input_ids[row_idx].tolist()
            start_pos = 0
            for pos in range(len(seq) - len(assistant_header) + 1):
                if seq[pos : pos + len(assistant_header)] == assistant_header:
                    start_pos = pos + len(assistant_header)
                    break
            labels[row_idx, :start_pos] = -100

        model_inputs["labels"] = labels
        model_inputs["pre_spatial"] = torch.stack([x["pre_spatial_tensor"] for x in batch], dim=0)
        model_inputs["post_spatial"] = torch.stack([x["post_spatial_tensor"] for x in batch], dim=0)
        model_inputs["gt_masks"] = torch.stack([x["gt_mask"] for x in batch], dim=0)
        model_inputs["is_refseg"] = torch.tensor([x["is_refseg"] for x in batch], dtype=torch.bool)
        model_inputs["is_nonempty"] = torch.tensor([x["is_nonempty"] for x in batch], dtype=torch.bool)
        model_inputs["is_recognition"] = torch.tensor([x["is_recognition"] for x in batch], dtype=torch.bool)
        model_inputs["is_longform"] = torch.tensor([x["is_longform"] for x in batch], dtype=torch.bool)
        model_inputs["post_modalities"] = [x["post_modality"] for x in batch]
        model_inputs["task_types"] = [x["task_type"] for x in batch]
        model_inputs["options"] = [x["options"] for x in batch]
        model_inputs["answers"] = [x["answer"] for x in batch]
        model_inputs["sample_ids"] = [x["sample_id"] for x in batch]
        return dict(model_inputs)
