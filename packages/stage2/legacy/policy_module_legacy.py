"""
Stage 2 policy LightningModule for token-based latent-action prediction.

Training loop:
1) Take a dict batch containing frame pairs + language.
2) Run frozen LAM (Stage 1) to produce discrete codes [B, code_seq_len].
3) Format codes into an action-token completion string.
4) Use Qwen3-VL processor to build multimodal inputs + prompt-masked labels.
5) Optimize standard LM loss on the completion tokens (Approach A).

This module is written to be unit-testable: the policy model, processor, and code
provider can be injected (use fakes for CPU tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch

from stage2.action_tokens import ActionTokenConfig
from stage2.constrained_decode import ActionTokenIds, make_prefix_allowed_tokens_fn
from stage2.online_lam import (
    LatentCodeProvider,
    frames_to_lam_video,
)
from stage2.policy_inputs import (
    ChatConfig,
    build_inputs_with_prompt_mask,
    build_prompt_inputs,
)


@dataclass
class PolicyOptimizerConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01


class PolicyTokenLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        policy_model: torch.nn.Module,
        processor: Any,
        code_provider: LatentCodeProvider,
        action_tokens: ActionTokenConfig,
        chat: Optional[ChatConfig] = None,
        optimizer: Optional[PolicyOptimizerConfig] = None,
        frames_to_images: Optional[Callable[[torch.Tensor], List[Any]]] = None,
        action_token_ids: Optional[ActionTokenIds] = None,
        train_teacher_forced_metrics_every_n_steps: int | None = None,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.processor = processor
        self.code_provider = code_provider
        self.action_tokens = action_tokens
        self.chat = chat or ChatConfig(system_prompt=None)
        self.optimizer_cfg = optimizer or PolicyOptimizerConfig()
        self.action_token_ids = action_token_ids
        self.train_teacher_forced_metrics_every_n_steps = (
            train_teacher_forced_metrics_every_n_steps
        )

        # Convert temporal frame tensors into image objects for the VLM processor.
        # In production this will likely return PIL Images; in tests we can inject a stub.
        self.frames_to_images = frames_to_images or (
            lambda frames: [object() for _ in range(frames.shape[0])]
        )

        if self.code_provider.codebook_size != self.action_tokens.codebook_size:
            raise ValueError(
                f"LAM codebook_size ({self.code_provider.codebook_size}) != "
                f"ActionTokenConfig ({self.action_tokens.codebook_size})"
            )
        if self.code_provider.code_seq_len != self.action_tokens.code_seq_len:
            raise ValueError(
                f"LAM code_seq_len ({self.code_provider.code_seq_len}) != "
                f"ActionTokenConfig ({self.action_tokens.code_seq_len})"
            )

    def _should_log_train_teacher_forced_metrics(self) -> bool:
        interval = self.train_teacher_forced_metrics_every_n_steps
        if interval is None:
            return False
        if interval <= 0:
            return False
        return (int(self.global_step) % int(interval)) == 0

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, codes, _frames, _instructions, inputs, outputs = self._loss_from_batch(
            batch
        )

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        if self._should_log_train_teacher_forced_metrics():
            self._log_gt_code_stats(stage="train", codes=codes)
            self._log_teacher_forced_code_metrics(
                stage="train", inputs=inputs, outputs=outputs
            )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, codes, frames, instructions, inputs, outputs = self._loss_from_batch(
            batch
        )

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self._log_gt_code_stats(stage="val", codes=codes)
        self._log_teacher_forced_code_metrics(
            stage="val", inputs=inputs, outputs=outputs
        )

        if (
            self.action_token_ids is not None
            and hasattr(self.policy_model, "generate")
            and batch_idx == 0
        ):
            pred_codes, gen_debug = self._predict_codes_with_debug(
                frames=frames, instructions=instructions
            )
            metrics = self._compute_generation_metrics(
                gt_codes=codes, pred_codes=pred_codes
            )
            self.log(
                "val/token_accuracy",
                metrics["token_accuracy"],
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/sequence_accuracy",
                metrics["sequence_accuracy"],
                prog_bar=False,
                sync_dist=True,
            )
            # Generation-parse health diagnostics.
            if gen_debug:
                start_frac = torch.tensor(
                    sum(1 for r in gen_debug if r.get("has_action_start"))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                end_frac = torch.tensor(
                    sum(1 for r in gen_debug if r.get("has_action_end"))
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                mean_codes = torch.tensor(
                    sum(int(r.get("num_codes_parsed", 0)) for r in gen_debug)
                    / float(len(gen_debug)),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.log(
                    "val/gen_has_action_start_frac",
                    start_frac,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "val/gen_has_action_end_frac",
                    end_frac,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    "val/gen_num_codes_parsed_mean",
                    mean_codes,
                    prog_bar=False,
                    sync_dist=True,
                )
            # Stash a small sample for visualization callbacks (rank0 only will use it).
            try:
                max_items = min(64, len(instructions), len(pred_codes))
                episode_id = (
                    batch.get("episode_id") if isinstance(batch, dict) else None
                )
                frame_idx = batch.get("frame_idx") if isinstance(batch, dict) else None
                self._last_val_sample = {
                    "frames": frames[:max_items].detach().cpu(),
                    "instructions": list(instructions[:max_items]),
                    "gt_codes": [
                        row.tolist() for row in codes[:max_items].detach().cpu()
                    ],
                    "pred_codes": [list(row) for row in pred_codes[:max_items]],
                    "gen_debug": gen_debug[:max_items],
                    "episode_id": list(episode_id[:max_items]) if episode_id else None,
                    "frame_idx": list(frame_idx[:max_items]) if frame_idx else None,
                }
            except Exception:
                self._last_val_sample = None

        return loss

    def _loss_from_batch(self, batch: Any) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[str],
        dict[str, torch.Tensor],
        Any,
    ]:
        if not isinstance(batch, dict):
            raise TypeError(
                "Expected dict batch with keys including frames and language."
            )

        frames = batch["frames"]
        language = batch.get("language")
        if not isinstance(language, list):
            raise TypeError("Expected batch['language'] to be a list[str]")
        instructions = [str(x) for x in language]

        video = frames_to_lam_video(frames)
        codes = self.code_provider.codes_from_video(video)  # [B, S]
        if codes.shape[0] != frames.shape[0]:
            raise ValueError("Batch size mismatch between frames and codes")

        targets = [self.action_tokens.format_target(row.tolist()) for row in codes]
        images = self.frames_to_images(frames)

        inputs = build_inputs_with_prompt_mask(
            processor=self.processor,
            images=images,
            instructions=instructions,
            targets=targets,
            chat=self.chat,
            device=self.device,
        )

        outputs = self.policy_model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return loss, codes, frames, instructions, inputs, outputs

    def _log_gt_code_stats(self, *, stage: str, codes: torch.Tensor) -> None:
        token_ids = self.action_token_ids
        if token_ids is None:
            return

        if codes.ndim != 2:
            return

        codebook_size = int(self.action_tokens.codebook_size)
        if codebook_size <= 0 or codebook_size > 64:
            return

        codes = codes.to(torch.long)
        if codes.numel() == 0:
            return

        # Overall code histogram across positions.
        flat = codes.reshape(-1)
        counts = torch.bincount(flat, minlength=codebook_size).to(torch.float32)
        on_step = stage == "train"
        on_epoch = not on_step
        for c in range(codebook_size):
            self.log(
                f"{stage}/gt_code_count/code{c}",
                counts[c],
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                reduce_fx="sum",
            )

        # Majority-token baseline per position (helps spot label imbalance).
        per_pos = []
        for pos in range(codes.shape[1]):
            pos_counts = torch.bincount(codes[:, pos], minlength=codebook_size).to(
                torch.float32
            )
            denom = pos_counts.sum().clamp(min=1.0)
            per_pos.append(pos_counts.max() / denom)
        if per_pos:
            baseline = torch.stack(per_pos).mean()
            self.log(
                f"{stage}/gt_majority_baseline_acc",
                baseline,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=int(codes.shape[0]),
            )

    def _log_teacher_forced_code_metrics(
        self, *, stage: str, inputs: dict[str, torch.Tensor], outputs: Any
    ) -> None:
        token_ids = self.action_token_ids
        if token_ids is None:
            return

        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            return

        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            return

        # Hugging Face causal LM convention: logits at position t predict label at t+1.
        # The model loss uses shifted labels/logits, so metrics must do the same.
        if labels.shape[1] < 2 or logits.shape[1] < 2:
            return
        shift_labels = labels[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()

        code_ids = torch.tensor(
            token_ids.action_code_ids, device=shift_labels.device, dtype=torch.long
        )
        if code_ids.numel() == 0:
            return

        # Identify positions where the supervision token is an action code token.
        active = shift_labels != -100
        is_code = (shift_labels.unsqueeze(-1) == code_ids.view(1, 1, -1)).any(
            dim=-1
        ) & active

        on_step = stage == "train"
        on_epoch = not on_step

        # Sanity-check: action wrapper tokens should usually be supervised once each.
        start_id = int(token_ids.action_start_id)
        end_id = int(token_ids.action_end_id)
        is_start = (shift_labels == start_id) & active
        is_end = (shift_labels == end_id) & active
        if shift_labels.shape[0] > 0:
            start_frac = is_start.any(dim=1).to(torch.float32).mean()
            end_frac = is_end.any(dim=1).to(torch.float32).mean()
            self.log(
                f"{stage}/label_has_action_start_frac",
                start_frac,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=int(shift_labels.shape[0]),
            )
            self.log(
                f"{stage}/label_has_action_end_frac",
                end_frac,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=int(shift_labels.shape[0]),
            )

        code_tokens_per_sample = is_code.sum(dim=1).to(torch.float32)
        expected = float(token_ids.code_seq_len)
        if shift_labels.shape[0] > 0:
            mismatch = (code_tokens_per_sample != expected).to(torch.float32).mean()
            self.log(
                f"{stage}/label_code_token_mismatch_frac",
                mismatch,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=int(shift_labels.shape[0]),
            )
            self.log(
                f"{stage}/label_code_tokens_per_sample_mean",
                code_tokens_per_sample.mean(),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                batch_size=int(shift_labels.shape[0]),
            )

        total = int(is_code.sum().item())
        if total <= 0:
            return

        # Restrict to the K action-code logits (K is small, e.g., 8).
        code_logits = shift_logits[..., code_ids]  # [B, T-1, K]
        selected_logits = code_logits[is_code]  # [N, K]

        # Teacher-forced predicted code index in [0, K-1].
        pred_idx = torch.argmax(selected_logits, dim=-1)
        pred_token_id = code_ids[pred_idx]
        gt_token_id = shift_labels[is_code]
        correct = (pred_token_id == gt_token_id).to(torch.float32).sum()
        acc = correct / float(total)

        # Entropy over the *code* distribution; low entropy indicates collapse.
        probs = torch.softmax(selected_logits, dim=-1)
        entropy = (-probs * (probs + 1e-9).log()).sum(dim=-1).mean()

        self.log(
            f"{stage}/tf_code_token_accuracy",
            acc,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
            prog_bar=(stage == "val"),
            batch_size=total,
        )
        self.log(
            f"{stage}/tf_code_token_entropy",
            entropy,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
            batch_size=total,
        )

        # Distribution diagnostics (counts per code index).
        k = int(code_ids.numel())
        pred_counts = torch.bincount(pred_idx, minlength=k).to(torch.float32)

        # Map gt token ids to indices 0..K-1 (safe because gt token ids are code tokens).
        gt_match = gt_token_id.unsqueeze(-1) == code_ids.view(1, -1)
        gt_idx = torch.argmax(gt_match.to(torch.long), dim=-1)
        gt_counts = torch.bincount(gt_idx, minlength=k).to(torch.float32)

        for i in range(k):
            self.log(
                f"{stage}/tf_pred_code_count/code{i}",
                pred_counts[i],
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                reduce_fx="sum",
            )
            self.log(
                f"{stage}/tf_gt_code_count/code{i}",
                gt_counts[i],
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                reduce_fx="sum",
            )

    @torch.no_grad()
    def _compute_generation_metrics(
        self, *, gt_codes: torch.Tensor, pred_codes: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        if gt_codes.ndim != 2:
            raise ValueError(
                f"Expected gt_codes [B, S], got shape {tuple(gt_codes.shape)}"
            )
        if len(pred_codes) != gt_codes.shape[0]:
            raise ValueError("Batch size mismatch between gt_codes and pred_codes")

        correct = 0
        total = 0
        seq_correct = 0

        for i in range(gt_codes.shape[0]):
            gt = gt_codes[i].tolist()
            pred = pred_codes[i][: len(gt)]
            if pred == gt:
                seq_correct += 1
            for p, g in zip(pred, gt, strict=True):
                total += 1
                if p == g:
                    correct += 1

        token_acc = 0.0 if total == 0 else (correct / total)
        seq_acc = 0.0 if gt_codes.shape[0] == 0 else (seq_correct / gt_codes.shape[0])
        return {
            "token_accuracy": torch.tensor(
                token_acc, device=self.device, dtype=torch.float32
            ),
            "sequence_accuracy": torch.tensor(
                seq_acc, device=self.device, dtype=torch.float32
            ),
        }

    @torch.no_grad()
    def _predict_codes(
        self, *, frames: torch.Tensor, instructions: list[str]
    ) -> list[list[int]]:
        pred_codes, _debug = self._predict_codes_with_debug(
            frames=frames, instructions=instructions
        )
        return pred_codes

    @torch.no_grad()
    def _predict_codes_with_debug(
        self, *, frames: torch.Tensor, instructions: list[str]
    ) -> tuple[list[list[int]], list[dict[str, Any]]]:
        images = self.frames_to_images(frames)
        prompt_inputs = build_prompt_inputs(
            processor=self.processor,
            images=images,
            instructions=instructions,
            chat=self.chat,
            device=self.device,
        )

        input_ids = prompt_inputs.get("input_ids")
        if input_ids is None:
            raise KeyError("processor output must include input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("processor output input_ids must be a 2D tensor")
        # `generate()` returns sequences that include the (padded) prompt for all items
        # followed by newly generated tokens. Always slice by the padded prompt length
        # to avoid misinterpreting prompt padding as generated output.
        prompt_len = int(input_ids.shape[1])

        token_ids = self.action_token_ids
        if token_ids is None:
            raise RuntimeError(
                "action_token_ids is required for constrained generation"
            )
        prefix_fn = make_prefix_allowed_tokens_fn(token_ids)

        # Constrained generations must include any separator tokens (e.g., spaces) that
        # are present in the supervised target format. Use a conservative budget.
        max_new = int(token_ids.code_seq_len) * 8 + 16
        generated = self.policy_model.generate(
            **prompt_inputs,
            max_new_tokens=max_new,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn,
        )

        # Map code token id -> code index
        code_id_to_index = {tid: i for i, tid in enumerate(token_ids.action_code_ids)}

        tok = getattr(self.processor, "tokenizer", None)
        decode = getattr(tok, "decode", None) if tok is not None else None

        results: list[list[int]] = []
        debug: list[dict[str, Any]] = []
        max_prompt_ids = 2048
        max_mask_ids = 2048
        max_suffix_ids = 256
        for i in range(generated.shape[0]):
            gen_suffix_t = generated[i, prompt_len:]
            gen_suffix = gen_suffix_t.tolist()
            # Parse strictly from the generated <ACTION> span to avoid accidentally
            # treating stray/padded tokens as codes.
            try:
                start_pos = gen_suffix.index(int(token_ids.action_start_id))
            except ValueError:
                start_pos = -1

            pred: list[int] = []
            has_end = False
            if start_pos >= 0:
                for t in gen_suffix[start_pos + 1 :]:
                    tid = int(t)
                    if tid == int(token_ids.action_end_id):
                        has_end = True
                        break
                    if tid in code_id_to_index:
                        if len(pred) < int(token_ids.code_seq_len):
                            pred.append(code_id_to_index[tid])
            if len(pred) < token_ids.code_seq_len:
                pred = pred + ([-1] * (token_ids.code_seq_len - len(pred)))
            results.append(pred)

            prompt_true_len: int | None = None
            attention_mask = prompt_inputs.get("attention_mask")
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
                # Include padding in the prompt decode so it is visible.
                try:
                    dbg["prompt_text_with_specials"] = str(
                        decode(input_ids[i].tolist(), skip_special_tokens=False)
                    )
                except Exception:
                    pass
                try:
                    dbg["generated_suffix_text_with_specials"] = str(
                        decode(gen_suffix, skip_special_tokens=False)
                    )
                except Exception:
                    pass
            # Always include raw ids for exact debugging; keep them bounded.
            try:
                prompt_ids = [int(x) for x in input_ids[i].tolist()]
                dbg["prompt_input_ids"] = prompt_ids[:max_prompt_ids]
                dbg["prompt_input_ids_truncated"] = len(prompt_ids) > max_prompt_ids
                if (
                    isinstance(attention_mask, torch.Tensor)
                    and attention_mask.ndim == 2
                ):
                    mask_ids = [int(x) for x in attention_mask[i].tolist()]
                    dbg["prompt_attention_mask"] = mask_ids[:max_mask_ids]
                    dbg["prompt_attention_mask_truncated"] = (
                        len(mask_ids) > max_mask_ids
                    )
                dbg["generated_suffix_ids"] = [
                    int(x) for x in gen_suffix[:max_suffix_ids]
                ]
                dbg["generated_suffix_ids_truncated"] = len(gen_suffix) > max_suffix_ids
            except Exception:
                pass
            debug.append(dbg)
        return results, debug

    @torch.no_grad()
    def _predict_freeform_text(
        self,
        *,
        frames: torch.Tensor,
        instructions: list[str],
        max_new_tokens: int = 32,
    ) -> list[str]:
        images = self.frames_to_images(frames)
        prompt_inputs = build_prompt_inputs(
            processor=self.processor,
            images=images,
            instructions=instructions,
            chat=self.chat,
            device=self.device,
        )

        input_ids = prompt_inputs.get("input_ids")
        if input_ids is None:
            raise KeyError("processor output must include input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("processor output input_ids must be a 2D tensor")
        prompt_len = int(input_ids.shape[1])

        generated = self.policy_model.generate(
            **prompt_inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        tok = getattr(self.processor, "tokenizer", None)
        decode = getattr(tok, "decode", None) if tok is not None else None
        if decode is None:
            return ["" for _ in range(int(generated.shape[0]))]

        texts: list[str] = []
        for i in range(int(generated.shape[0])):
            suffix_ids = generated[i, prompt_len:].tolist()
            texts.append(str(decode(suffix_ids, skip_special_tokens=False)))
        return texts

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        return optimizer
