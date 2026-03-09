from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from stage2.policy_inputs import ChatConfig, build_inputs_with_prompt_mask


@dataclass
class _FakeBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def items(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }.items()


class FakeProcessor:
    """
    Minimal processor stub:
    - apply_chat_template returns the concatenated text segments
    - __call__ returns padded input_ids/attention_mask with image-token contribution
    """

    def __init__(self, image_tokens_per_image: int = 5):
        self.image_tokens_per_image = image_tokens_per_image

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        parts: List[str] = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("<image>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join([p for p in parts if p])

    def __call__(
        self,
        *,
        text: Sequence[str],
        images: Sequence[Any],
        return_tensors: str,
        padding: bool,
    ):
        assert return_tensors == "pt"
        assert padding is True

        lengths: List[int] = []
        for t in text:
            # token length = num words + image tokens for each "<image>" marker
            num_words = len([w for w in t.split(" ") if w])
            num_images = t.count("<image>")
            lengths.append(num_words + num_images * self.image_tokens_per_image)

        max_len = max(lengths)
        input_ids = torch.zeros((len(text), max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)

        for i, length in enumerate(lengths):
            input_ids[i, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[i, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_build_inputs_with_prompt_mask_masks_prompt_and_padding():
    processor = FakeProcessor(image_tokens_per_image=3)
    images = [object(), object()]
    instructions = ["pick up block", "push button"]
    targets = [
        "<ACTION> <ACT_0> <ACT_1> <ACT_2> <ACT_3> </ACTION>",
        "<ACTION> <ACT_7> <ACT_7> <ACT_7> <ACT_7> </ACTION>",
    ]

    batch = build_inputs_with_prompt_mask(
        processor=processor,
        images=images,
        instructions=instructions,
        targets=targets,
        chat=ChatConfig(system_prompt="You are a robot."),
        device=torch.device("cpu"),
    )

    assert "input_ids" in batch and "attention_mask" in batch and "labels" in batch
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.shape == labels.shape
    assert attention_mask.shape == labels.shape

    # Prompt lens computed using processor(prompt_only, images=...)
    prompt_texts = [
        processor.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a robot."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": instructions[i]},
                    ],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(len(images))
    ]
    prompt_inputs = processor(
        text=prompt_texts, images=images, return_tensors="pt", padding=True
    )
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

    for i, pl in enumerate(prompt_lens):
        pl = int(pl)
        # Prompt is masked.
        assert torch.all(labels[i, :pl] == -100)
        # Non-padding after prompt should be unmasked (equal to input ids).
        nonpad = attention_mask[i].bool()
        completion_positions = torch.arange(labels.shape[1]) >= pl
        check_positions = nonpad & completion_positions
        assert torch.all(labels[i, check_positions] == input_ids[i, check_positions])
        # Padding is masked.
        assert torch.all(labels[i, ~nonpad] == -100)
