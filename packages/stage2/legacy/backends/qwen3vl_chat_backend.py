from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.backends.interfaces import (
    BackendMode,
    Stage2Batch,
    LatentOutput,
    LossOutput,
)
from stage2.constrained_decode import ActionTokenIds, make_prefix_allowed_tokens_fn
from stage2.image_adapters import oxe_first_frames_to_pil
from stage2.legacy.qwen3vl_setup import prepare_action_token_training
from stage2.policy_inputs import (
    ChatConfig,
    build_inputs_with_prompt_mask,
    build_prompt_inputs,
)


def _infer_between_action_token_ids(
    *,
    tokenizer: Any,
    action_cfg: ActionTokenConfig,
    token_id_map: dict[str, int],
) -> list[int]:
    """
    Copy of `scripts/4_train_stage2_policy.py:infer_between_action_token_ids`.

    Action targets include separators (currently spaces). Constrained decoding must
    allow these token ids, or generation tokenization can mismatch supervised targets.
    """
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer must implement encode(text, add_special_tokens=...)")

    example = action_cfg.format_target([0] * int(action_cfg.code_seq_len))
    ids = [int(x) for x in encode(example, add_special_tokens=False)]

    special: set[int] = {
        int(token_id_map[action_cfg.action_start]),
        int(token_id_map[action_cfg.action_end]),
    }
    for i in range(int(action_cfg.codebook_size)):
        special.add(int(token_id_map[action_cfg.token_fmt.format(i=i)]))

    between = [tid for tid in ids if int(tid) not in special]
    out: list[int] = []
    seen: set[int] = set()
    for tid in between:
        if tid in seen:
            continue
        seen.add(tid)
        out.append(int(tid))

    if not out:
        raise RuntimeError(
            "Could not infer any between-action token ids from the target string. "
            "This would make constrained decoding mismatch the supervised targets."
        )
    return out


@dataclass
class Qwen3VLChatBackendConfig:
    model_name: str
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str | None = None
    chat: ChatConfig = ChatConfig(system_prompt=None)
    action_tokens: ActionTokenConfig = ActionTokenConfig()


class Qwen3VLChatActionTokenBackend(torch.nn.Module):
    """
    Qwen3-VL / Cosmos-Reason2 backend that matches HLRP Stage 2 behavior:
    - Uses `processor.apply_chat_template(...)` with image+text messages
    - Prompt-masked LM loss on action-token completion
    - Constrained decoding via `prefix_allowed_tokens_fn`
    """

    def __init__(
        self,
        *,
        config: Qwen3VLChatBackendConfig,
        policy_model: torch.nn.Module | None = None,
        processor: Any | None = None,
        frames_to_images: Callable[[torch.Tensor], List[Any]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.policy_model = policy_model
        self.processor = processor
        self.frames_to_images = frames_to_images or oxe_first_frames_to_pil

        self.codebook_size = int(self.cfg.action_tokens.codebook_size)
        self.code_seq_len = int(self.cfg.action_tokens.code_seq_len)

        self._action_token_ids: ActionTokenIds | None = None

    @staticmethod
    def _extract_frames(batch: Stage2Batch) -> torch.Tensor:
        streams = batch.image_streams
        if streams is None:
            raise ValueError("batch.image_streams is required")
        if "observation.images.rgb" not in streams:
            raise KeyError(
                "batch.image_streams must include key 'observation.images.rgb'"
            )
        return streams["observation.images.rgb"]

    @staticmethod
    def _extract_instructions(batch: Stage2Batch) -> list[str]:
        if batch.task_text is None:
            raise ValueError("batch.task_text is required")
        return [str(x) for x in batch.task_text]

    def setup(self, *, device: torch.device) -> None:
        if self.policy_model is None or self.processor is None:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

            self.policy_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.cfg.model_name,
                torch_dtype=self.cfg.torch_dtype,
                attn_implementation=self.cfg.attn_implementation,
            )
            self.policy_model.train()
            self.processor = Qwen3VLProcessor.from_pretrained(self.cfg.model_name)

        self.policy_model.to(device)

        token_id_map = prepare_action_token_training(
            model=self.policy_model,
            processor=self.processor,
            action_tokens=self.cfg.action_tokens,
        )

        between_ids = _infer_between_action_token_ids(
            tokenizer=self.processor.tokenizer,
            action_cfg=self.cfg.action_tokens,
            token_id_map=token_id_map,
        )

        self._action_token_ids = ActionTokenIds(
            action_start_id=int(token_id_map[self.cfg.action_tokens.action_start]),
            action_end_id=int(token_id_map[self.cfg.action_tokens.action_end]),
            action_code_ids=[
                int(token_id_map[self.cfg.action_tokens.token_fmt.format(i=i)])
                for i in range(int(self.cfg.action_tokens.codebook_size))
            ],
            between_token_ids=between_ids,
            eos_token_id=int(getattr(self.processor.tokenizer, "eos_token_id", 0)),
            code_seq_len=int(self.cfg.action_tokens.code_seq_len),
        )

    def _require_ready(self) -> tuple[torch.device, ActionTokenIds]:
        if self.policy_model is None or self.processor is None:
            raise RuntimeError("Backend not initialized. Call setup(device=...) first.")
        if self._action_token_ids is None:
            raise RuntimeError(
                "Action token ids not initialized. Call setup(device=...) first."
            )
        try:
            device = next(self.policy_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return device, self._action_token_ids

    @property
    def action_token_ids(self) -> ActionTokenIds:
        _device, ids = self._require_ready()
        return ids

    def loss_from_batch(self, batch: Stage2Batch, *, mode: BackendMode) -> LossOutput:
        if mode is not BackendMode.CODES:
            raise NotImplementedError(
                f"{type(self).__name__} only supports mode={BackendMode.CODES.value!r}"
            )
        device, _token_ids = self._require_ready()

        frames = self._extract_frames(batch)
        instructions = self._extract_instructions(batch)
        if batch.target_codes is None:
            raise ValueError(
                "batch.target_codes is required for code loss computation."
            )
        codes = batch.target_codes.to(torch.long)
        if codes.ndim != 2 or codes.shape[1] != self.code_seq_len:
            raise ValueError(
                f"Expected codes [B, {self.code_seq_len}], got {tuple(codes.shape)}"
            )

        targets = [self.cfg.action_tokens.format_target(row.tolist()) for row in codes]
        images = self.frames_to_images(frames)

        inputs = build_inputs_with_prompt_mask(
            processor=self.processor,
            images=images,
            instructions=instructions,
            targets=targets,
            chat=self.cfg.chat,
            device=device,
        )

        outputs = self.policy_model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return LossOutput(
            loss=loss, metrics={"loss": float(loss.detach().cpu().item())}
        )

    @torch.no_grad()
    def latent_from_batch(
        self, batch: Stage2Batch, *, mode: BackendMode
    ) -> LatentOutput:
        if mode is not BackendMode.CODES:
            raise NotImplementedError(
                f"{type(self).__name__} only supports mode={BackendMode.CODES.value!r}"
            )
        device, token_ids = self._require_ready()

        frames = self._extract_frames(batch)
        instructions = self._extract_instructions(batch)
        images = self.frames_to_images(frames)
        prompt_inputs = build_prompt_inputs(
            processor=self.processor,
            images=images,
            instructions=instructions,
            chat=self.cfg.chat,
            device=device,
        )

        input_ids = prompt_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("processor output input_ids must be a 2D tensor")
        prompt_len = int(input_ids.shape[1])

        prefix_fn = make_prefix_allowed_tokens_fn(token_ids)
        max_new = int(token_ids.code_seq_len) * 8 + 16
        generated = self.policy_model.generate(
            **prompt_inputs,
            max_new_tokens=max_new,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
        )

        code_id_to_index = {
            int(tid): i for i, tid in enumerate(token_ids.action_code_ids)
        }
        tok = getattr(self.processor, "tokenizer", None)
        decode = getattr(tok, "decode", None) if tok is not None else None

        b = int(generated.shape[0])
        tokens_out = torch.full((b, int(token_ids.code_seq_len)), -1, dtype=torch.long)
        suffix_ids_out: list[list[int]] = []
        debug: list[dict[str, Any]] = []

        attention_mask = prompt_inputs.get("attention_mask")
        for i in range(int(generated.shape[0])):
            gen_suffix_t = generated[i, prompt_len:]
            gen_suffix = [int(x) for x in gen_suffix_t.tolist()]
            suffix_ids_out.append(gen_suffix)

            try:
                start_pos = gen_suffix.index(int(token_ids.action_start_id))
            except ValueError:
                start_pos = -1

            pred: list[int] = []
            has_end = False
            if start_pos >= 0:
                for tid in gen_suffix[start_pos + 1 :]:
                    if tid == int(token_ids.action_end_id):
                        has_end = True
                        break
                    if tid in code_id_to_index and len(pred) < int(
                        token_ids.code_seq_len
                    ):
                        pred.append(int(code_id_to_index[tid]))

            if len(pred) < int(token_ids.code_seq_len):
                pred = pred + ([-1] * (int(token_ids.code_seq_len) - len(pred)))
            tokens_out[i] = torch.tensor(pred, dtype=torch.long)

            prompt_true_len: int | None = None
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                try:
                    prompt_true_len = int(attention_mask[i].sum().item())
                except Exception:
                    prompt_true_len = None

            dbg: dict[str, Any] = {
                "has_action_start": start_pos >= 0,
                "has_action_end": has_end,
                "num_codes_parsed": int(sum(1 for x in pred if int(x) >= 0)),
                "prompt_padded_len": int(prompt_len),
                "prompt_true_len": (
                    int(prompt_true_len) if prompt_true_len is not None else None
                ),
            }
            if decode is not None:
                try:
                    dbg["generated_suffix_text_with_specials"] = str(
                        decode(gen_suffix, skip_special_tokens=False)
                    )
                except Exception:
                    pass
            debug.append(dbg)

        return LatentOutput(
            logits=None,
            tokens=tokens_out,
            vector=None,
            meta={
                "generated_suffix_token_ids": suffix_ids_out,
                "parse_debug": debug,
            },
        )
