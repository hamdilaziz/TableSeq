from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from tableseq import TableSeqModel
from tableseq.modeling import extract_sequences
from tableseq.training.box_metrics import (
    average_metric_dicts,
    build_coordinate_vocab_masks,
    compute_box_metrics_from_strings,
    coordinate_token_accuracy_from_logits,
)
from tableseq.training.fast_metrics import FastTableMetrics
from tableseq.training.metrics import clean_tableseq_html
from tableseq.training.noise import TokenSubstitutionNoise


@dataclass
class TableSeqTrainingConfig:
    output_dir: str = "outputs/tableseq_finetune"
    max_epochs: int = 1
    max_steps: Optional[int] = None
    max_length: int = 2048

    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = True

    teacher_forcing_error_rate: float = 0.02
    token_substitution_path: Optional[str] = None

    # Step-based training control. This is more useful than epoch-based control
    # when one epoch contains hundreds of thousands of samples.
    log_every_n_steps: int = 20
    save_every_n_steps: int = 2000
    eval_every_n_steps: int = 2000
    generate_samples_every_n_steps: int = 2000
    num_prediction_samples: int = 3

    # Validation subsets. None means full validation loader.
    max_valid_batches_for_loss: Optional[int] = 100
    max_valid_batches_for_generation: Optional[int] = 50
    compute_valid_loss: bool = False
    compute_valid_teds: bool = False
    compute_valid_steds: bool = True
    compute_valid_struct_seq: bool = True
    compute_valid_box_metrics: bool = True
    metric_for_best: str = "valid_steds"

    validate_at_start: bool = False
    validate_at_end: bool = True
    save_at_end: bool = True

    coord_loss_weight: float = 1.0

    # Resume behavior. If auto_resume=True and output_dir/checkpoints/last.pt
    # exists, the trainer resumes from that checkpoint. This overrides the
    # transfer checkpoint already loaded by TableSeqModel.
    auto_resume: bool = True
    resume_checkpoint_path: Optional[str] = None
    resume_optimizer: bool = True
    resume_scheduler: bool = True
    resume_scaler: bool = True
    allow_resume_data_config_mismatch: bool = False

    # Coordinates / layout monitoring. These metrics are computed only if the
    # tokenizer and labels contain <x_i>/<y_j> tokens.
    compute_train_coord_token_acc: bool = True
    show_coordinates_in_samples: bool = True


def pad_1d_sequences(sequences: list[Tensor], max_length: int, pad_token_id: int) -> Tensor:
    if len(sequences) == 0:
        raise ValueError("Cannot pad an empty sequence list.")
    device = sequences[0].device
    out = torch.full((len(sequences), max_length), fill_value=pad_token_id, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        seq = seq[:max_length]
        out[i, : seq.numel()] = seq
    return out


def build_teacher_forcing_inputs(
    labels: Tensor,
    labels_len: list[int],
    token_prompts: list[Tensor],
    pad_token_id: int,
    max_length: int,
) -> tuple[Tensor, Tensor]:
    """Build decoder input ids and shifted target labels.

    The batch labels are expected to contain ``<s> <html> target </s>``.
    The target is shifted by one token. Padding tokens are ignored by the
    decoder loss.

    The start token is ignored before shifting. With the default one-token
    ``<html>`` prompt, the prompt remains the first supervised target token.
    """
    y = labels.clone()
    for i, prompt in enumerate(token_prompts):
        prompt_len = int(prompt.numel())
        y[i, :prompt_len] = pad_token_id

    input_sequences = []
    target_sequences = []
    for i, seq_len in enumerate(labels_len):
        seq_len = min(int(seq_len), max_length)
        input_sequences.append(labels[i, :seq_len])
        target_sequences.append(y[i, 1:seq_len])

    input_ids = pad_1d_sequences(input_sequences, max_length=max_length, pad_token_id=pad_token_id)
    target_labels = pad_1d_sequences(target_sequences, max_length=max_length, pad_token_id=pad_token_id)
    return input_ids, target_labels


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


class TableSeqTrainer:
    def __init__(
        self,
        model: TableSeqModel,
        train_loader,
        valid_loader=None,
        config: Optional[TableSeqTrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config or TableSeqTrainingConfig()

        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=self.config.use_amp and self.device.type == "cuda")

        self.global_step = 0
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_metric: Optional[float] = None
        self.last_log_time = time.time()
        self.last_log_step = 0

        self.output_dir = Path(self.config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.data_config = self._extract_data_config()
        self.checkpoint_data_config = self._build_checkpoint_data_config()
        self.checkpoint_training_config = self._build_checkpoint_training_config()

        self.fast_metrics = FastTableMetrics()
        self.coord_x_vocab_mask, self.coord_y_vocab_mask = build_coordinate_vocab_masks(
            self.model.decoder.tokenizer
        )
        print(
            "Coordinate tokens in tokenizer: "
            f"x={int(self.coord_x_vocab_mask.sum().item())}, "
            f"y={int(self.coord_y_vocab_mask.sum().item())}."
        )
        self.noise_injector: Optional[TokenSubstitutionNoise] = None
        if self.config.token_substitution_path is not None:
            self.noise_injector = TokenSubstitutionNoise.from_json(
                self.config.token_substitution_path,
                self.model.decoder.tokenizer,
            )
            print(
                "Loaded token substitution noise: "
                f"{len(self.noise_injector.substitution_dict)} source tokens, "
                f"skipped_sources={getattr(self.noise_injector, 'skipped_sources', 0)}, "
                f"skipped_replacements={getattr(self.noise_injector, 'skipped_replacements', 0)}."
            )

        self.maybe_resume_training()
        self._write_config()

    def _extract_data_config(self) -> Optional[dict[str, Any]]:
        for loader in (self.train_loader, self.valid_loader):
            dataset = getattr(loader, "dataset", None) if loader is not None else None
            config = getattr(dataset, "config", None)
            if config is not None:
                try:
                    return asdict(config)
                except TypeError:
                    if isinstance(config, dict):
                        return dict(config)
        return None

    def _build_checkpoint_data_config(self) -> Optional[dict[str, Any]]:
        """Keep reproducibility metadata without embedding machine-specific paths."""
        if not isinstance(self.data_config, dict):
            return None
        keys = (
            "label_file",
            "mean",
            "std",
            "resize_factor",
            "coordinate_labels_in_original_image_space",
            "strict_coordinate_vocabulary",
            "normalize",
            "image_padding_value",
            "default_prompt",
            "max_height",
            "max_width",
        )
        return {key: self.data_config.get(key) for key in keys if key in self.data_config}

    def _build_checkpoint_training_config(self) -> dict[str, Any]:
        """Remove machine-specific paths from metadata intended to be shared."""
        config = asdict(self.config)
        for key in ("output_dir", "token_substitution_path", "resume_checkpoint_path"):
            config.pop(key, None)
        return config

    def _write_config(self) -> None:
        path = self.output_dir / "training_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

        if self.data_config is not None:
            data_path = self.output_dir / "data_config.json"
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(self.data_config, f, indent=2, ensure_ascii=False)

    def _get_resume_checkpoint_path(self) -> Path:
        if self.config.resume_checkpoint_path is not None:
            return Path(self.config.resume_checkpoint_path)
        return self.checkpoint_dir / "last.pt"

    def maybe_resume_training(self) -> None:
        """
        Resume an interrupted training run if a last checkpoint exists.

        The normal initialization order is:
            1. TableSeqModel loads transfer/pretrained weights from
               config["checkpoints_path"], if provided.
            2. TableSeqTrainer checks output_dir/checkpoints/last.pt.
            3. If last.pt exists, it restores model, optimizer, scaler,
               scheduler, global_step, and best_metric.

        Therefore, a resume checkpoint always has priority over the transfer
        checkpoint, while a new training run still starts from the transfer
        checkpoint.
        """
        if not self.config.auto_resume:
            print("[resume] auto_resume=False. Starting from the current model weights.")
            return

        resume_path = self._get_resume_checkpoint_path()

        if not resume_path.exists():
            print(
                f"[resume] No previous training checkpoint found at {resume_path}. "
                "Starting a new training run from the current model weights."
            )
            return

        print(f"[resume] Found previous training checkpoint: {resume_path}")
        self.load_training_checkpoint(resume_path)

    def _validate_resume_data_config(self, checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
        saved = checkpoint.get("data_config")
        current = self.data_config
        if not isinstance(saved, dict) or not isinstance(current, dict):
            return

        keys = (
            "resize_factor",
            "coordinate_labels_in_original_image_space",
            "max_height",
            "max_width",
            "label_file",
        )
        mismatches = {
            key: (saved.get(key), current.get(key))
            for key in keys
            if saved.get(key) != current.get(key)
        }
        if not mismatches:
            return

        details = ", ".join(
            f"{key}: saved={old!r}, current={new!r}"
            for key, (old, new) in mismatches.items()
        )
        message = (
            f"Resume data configuration differs from {checkpoint_path}: {details}. "
            "Changing image or coordinate geometry while resuming can corrupt training. "
            "Use a new output directory for a new resolution experiment."
        )
        if self.config.allow_resume_data_config_mismatch:
            print(f"[resume] WARNING: {message}")
        else:
            raise ValueError(message)

    def load_training_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """
        Load a full training checkpoint.

        Expected format:
            {
                "global_step": int,
                "epoch": int,
                "encoder_state_dict": ...,
                "decoder_state_dict": ...,
                "optimizer_state_dict": ...,
                "scaler_state_dict": ...,
                "scheduler_state_dict": ...,
                "best_metric": float,
                ...
            }
        """
        checkpoint_path = Path(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self._validate_resume_data_config(checkpoint, checkpoint_path)

        if "encoder_state_dict" not in checkpoint or "decoder_state_dict" not in checkpoint:
            raise KeyError(
                f"{checkpoint_path} is not a full TableSeq training checkpoint. "
                "It must contain 'encoder_state_dict' and 'decoder_state_dict'."
            )

        encoder_info = self.model.encoder.load_state_dict(
            checkpoint["encoder_state_dict"],
            strict=True,
        )
        decoder_info = self.model.decoder.load_state_dict(
            checkpoint["decoder_state_dict"],
            strict=True,
        )

        if self.config.resume_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[resume] Optimizer state restored.")
        else:
            print("[resume] Optimizer state not restored.")

        if self.scheduler is not None and self.config.resume_scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("[resume] Scheduler state restored.")
        elif self.scheduler is not None:
            print("[resume] Scheduler state not restored.")

        if self.config.resume_scaler and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print("[resume] AMP scaler state restored.")
            except Exception as exc:
                print(f"[resume] AMP scaler state could not be restored: {exc}")
        else:
            print("[resume] AMP scaler state not restored.")

        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.current_epoch = self.start_epoch

        best_metric = checkpoint.get("best_metric", None)
        self.best_metric = None if best_metric is None else float(best_metric)

        # Avoid an artificial spike in steps/sec after resuming.
        self.last_log_time = time.time()
        self.last_log_step = self.global_step

        metric_name = checkpoint.get("best_metric_name", self.config.metric_for_best)

        print(
            "[resume] Restored training state: "
            f"global_step={self.global_step}, "
            f"start_epoch={self.start_epoch}, "
            f"best_{metric_name}={self.best_metric}."
        )
        print(f"[resume] Encoder load info: {encoder_info}")
        print(f"[resume] Decoder load info: {decoder_info}")

        return checkpoint

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return str(value)
            if value != 0.0 and abs(value) < 1e-3:
                return f"{value:.2e}"
            return f"{value:.4f}"
        return str(value)

    def log_metrics(self, metrics: dict[str, Any], phase: str) -> None:
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "step": self.global_step,
            **metrics,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        printable = ", ".join(
            f"{k}={self._format_metric_value(v)}"
            for k, v in record.items()
            if k not in {"time", "phase"}
        )
        print(f"[{phase}] {printable}")

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch["imgs"].to(self.device, non_blocking=True)
        image_attention_mask = batch.get("image_attention_mask")
        if image_attention_mask is not None:
            image_attention_mask = image_attention_mask.to(self.device, non_blocking=True)

        labels = batch["labels"].to(self.device, non_blocking=True)
        labels_len = batch["labels_len"]
        if torch.is_tensor(labels_len):
            labels_len = labels_len.detach().cpu().tolist()
        labels_len = [int(x) for x in labels_len]

        token_prompts = []
        for prompt in batch["token_prompt"]:
            if torch.is_tensor(prompt):
                token_prompts.append(prompt.to(self.device, non_blocking=True))
            else:
                token_prompts.append(torch.tensor(prompt, dtype=torch.long, device=self.device))

        return {
            "images": images,
            "image_attention_mask": image_attention_mask,
            "labels": labels,
            "labels_len": labels_len,
            "token_prompts": token_prompts,
        }

    def compute_loss_on_batch(self, batch: dict[str, Any], train: bool) -> dict[str, Any]:
        prepared = self._prepare_batch(batch)
        images = prepared["images"]
        image_attention_mask = prepared["image_attention_mask"]
        labels = prepared["labels"]
        labels_len = prepared["labels_len"]
        token_prompts = prepared["token_prompts"]

        tokenizer = self.model.decoder.tokenizer
        pad_token_id = int(tokenizer.pad_token_id)
        clean_labels = labels

        if self.noise_injector is not None and train:
            input_source_labels = self.noise_injector(
                labels=clean_labels,
                labels_len=labels_len,
                token_prompts=token_prompts,
                error_rate=self.config.teacher_forcing_error_rate,
            )
            noise_stats = dict(self.noise_injector.last_stats)
        else:
            input_source_labels = clean_labels
            noise_stats = {"eligible": 0, "changed": 0}

        input_ids, _ = build_teacher_forcing_inputs(
            labels=input_source_labels,
            labels_len=labels_len,
            token_prompts=token_prompts,
            pad_token_id=pad_token_id,
            max_length=self.config.max_length,
        )
        _, target_labels = build_teacher_forcing_inputs(
            labels=clean_labels,
            labels_len=labels_len,
            token_prompts=token_prompts,
            pad_token_id=pad_token_id,
            max_length=self.config.max_length,
        )

        decoder_attention_mask = input_ids.ne(pad_token_id).long()
        non_pad_tokens = int(target_labels.ne(pad_token_id).sum().item())

        amp_enabled = self.config.use_amp and self.device.type == "cuda"
        with autocast(enabled=amp_enabled):
            decoder_output = self.model(
                images=images,
                input_ids=input_ids,
                labels=None,
                attention_mask=decoder_attention_mask,
                image_attention_mask=image_attention_mask,
                use_structure_bias=False,
                use_cache=False,
            )

            logits = decoder_output["logits"] if isinstance(decoder_output, dict) else decoder_output.logits

            vocab_size = logits.size(-1)

            loss_per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_labels.reshape(-1),
                ignore_index=pad_token_id,
                reduction="none",
            ).view_as(target_labels)

            valid_mask = target_labels.ne(pad_token_id)

            weights = torch.ones_like(loss_per_token)

            if self.config.coord_loss_weight != 1.0:
                x_mask = self.coord_x_vocab_mask.to(target_labels.device)
                y_mask = self.coord_y_vocab_mask.to(target_labels.device)

                if x_mask.numel() < vocab_size:
                    pad = torch.zeros(vocab_size - x_mask.numel(), dtype=torch.bool, device=target_labels.device)
                    x_mask = torch.cat([x_mask, pad], dim=0)
                    y_mask = torch.cat([y_mask, pad], dim=0)

                x_mask = x_mask[:vocab_size]
                y_mask = y_mask[:vocab_size]

                safe_labels = target_labels.clamp(min=0, max=vocab_size - 1)
                coord_targets = x_mask[safe_labels] | y_mask[safe_labels]

                weights = torch.where(
                    coord_targets,
                    torch.full_like(weights, float(self.config.coord_loss_weight)),
                    weights,
                )

            weighted_mask = weights * valid_mask.float()

            ce_loss = (loss_per_token * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            loss = ce_loss
        detached_logits = logits.detach()
        detached_labels = target_labels.detach()
        token_predictions = torch.argmax(detached_logits, dim=-1)
        token_mask = detached_labels.ne(pad_token_id)
        token_count = int(token_mask.sum().item())
        token_correct = int(token_predictions.eq(detached_labels).logical_and(token_mask).sum().item())
        token_acc = float(token_correct / token_count) if token_count else 0.0

        out: dict[str, Any] = {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "token_acc": token_acc,
            "token_count": token_count,
            "token_correct": token_correct,
            "non_pad_tokens": non_pad_tokens,
            "noise_eligible": int(noise_stats.get("eligible", 0)),
            "noise_changed": int(noise_stats.get("changed", 0)),
        }

        if self.config.compute_train_coord_token_acc:
            out.update(
                coordinate_token_accuracy_from_logits(
                    logits=detached_logits,
                    labels=detached_labels,
                    x_vocab_mask=self.coord_x_vocab_mask,
                    y_vocab_mask=self.coord_y_vocab_mask,
                )
            )

        return out

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        start = time.time()

        out = self.compute_loss_on_batch(batch, train=True)
        loss = out["loss"]
        self.scaler.scale(loss).backward()

        grad_norm = 0.0
        if self.config.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            grad_norm = float(grad_norm_tensor.detach().cpu())

        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        elapsed = max(time.time() - start, 1e-6)
        tokens_per_sec = float(out["non_pad_tokens"] / elapsed)

        stats = {
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(out["ce_loss"].detach().cpu()),
            "token_acc": float(out["token_acc"]),
            "grad_norm": grad_norm,
            "lr": get_lr(self.optimizer),
            "tokens_per_sec": tokens_per_sec,
            "noise_changed": float(out["noise_changed"]),
            "noise_eligible": float(out["noise_eligible"]),
        }

        for key in (
            "coord_token_acc",
            "x_token_acc",
            "y_token_acc",
            "coord_token_count",
            "x_token_count",
            "y_token_count",
        ):
            if key in out:
                stats[key] = float(out[key])

        return stats

    @torch.no_grad()
    def validate_loss(self) -> dict[str, float]:
        if self.valid_loader is None:
            return {}
        if not self.config.compute_valid_loss:
            return {}

        self.model.eval()
        loss_values: list[float] = []
        ce_loss_values: list[float] = []
        totals = {
            "token_count": 0.0,
            "token_correct": 0.0,
            "coord_token_count": 0.0,
            "coord_token_correct": 0.0,
            "x_token_count": 0.0,
            "x_token_correct": 0.0,
            "y_token_count": 0.0,
            "y_token_correct": 0.0,
        }

        max_batches = self.config.max_valid_batches_for_loss
        total = len(self.valid_loader) if max_batches is None else min(len(self.valid_loader), max_batches)
        pbar = tqdm(total=total, desc="Validation loss", leave=False)

        for batch_idx, batch in enumerate(self.valid_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            out = self.compute_loss_on_batch(batch, train=False)
            loss_values.append(float(out["loss"].detach().cpu()))
            ce_loss_values.append(float(out["ce_loss"].detach().cpu()))
            for key in totals:
                if key in out:
                    totals[key] += float(out[key])

            token_acc = totals["token_correct"] / totals["token_count"] if totals["token_count"] else 0.0
            postfix = {
                "loss": f"{float(np.mean(loss_values)):.4f}",
                "acc": f"{token_acc:.4f}",
                "n": len(loss_values),
            }
            if totals["coord_token_count"]:
                postfix["coord_acc"] = (
                    f"{totals['coord_token_correct'] / totals['coord_token_count']:.4f}"
                )
            pbar.update(1)
            pbar.set_postfix(**postfix)

        pbar.close()
        if not loss_values:
            return {}

        metrics: dict[str, float] = {
            "valid_loss": float(np.mean(loss_values)),
            "valid_ce_loss": float(np.mean(ce_loss_values)),
            "valid_token_count": float(totals["token_count"]),
        }
        if totals["token_count"]:
            metrics["valid_token_acc"] = float(totals["token_correct"] / totals["token_count"])

        for prefix in ("coord_token", "x_token", "y_token"):
            count = totals[f"{prefix}_count"]
            correct = totals[f"{prefix}_correct"]
            metrics[f"valid_{prefix}_count"] = float(count)
            if count:
                metrics[f"valid_{prefix}_acc"] = float(correct / count)

        return metrics

    @torch.no_grad()
    def validate_generation(self) -> dict[str, float]:
        if self.valid_loader is None:
            return {}

        if not any(
            (
                self.config.compute_valid_struct_seq,
                self.config.compute_valid_steds,
                self.config.compute_valid_teds,
                self.config.compute_valid_box_metrics,
            )
        ):
            return {}

        self.model.eval()
        tokenizer = self.model.decoder.tokenizer

        # Tree-based S-TEDS: structure only, cached GT tree.
        steds_scores: list[float] = []

        # Very fast proxy based on linearized table tags.
        struct_seq_scores: list[float] = []

        # Full TEDS, slower because it includes text.
        teds_scores: list[float] = []

        box_metric_dicts: list[dict[str, float]] = []

        generated_lengths: list[int] = []

        max_batches = self.config.max_valid_batches_for_generation
        total = len(self.valid_loader) if max_batches is None else min(len(self.valid_loader), max_batches)
        pbar = tqdm(total=total, desc="Validation generation", leave=True)

        processed_batches = 0

        for batch_idx, batch in enumerate(self.valid_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["imgs"].to(self.device, non_blocking=True)

            image_attention_mask = batch.get("image_attention_mask")
            if image_attention_mask is not None:
                image_attention_mask = image_attention_mask.to(self.device, non_blocking=True)

            batch_size = images.size(0)

            input_ids, _ = self.model.build_input_ids_from_prompt(
                "<html>",
                batch_size=batch_size,
                device=self.device,
            )

            output = self.model.generate(
                images=images,
                input_ids=input_ids,
                image_attention_mask=image_attention_mask,
                max_length=self.config.max_length,
                use_cache=True,
                use_structure_bias=False,
                return_dict_in_generate=True,
            )

            sequences = extract_sequences(output)

            predictions = [
                tokenizer.decode(seq, skip_special_tokens=False)
                for seq in sequences
            ]

            references = batch["raw_labels"]
            geometries = batch.get("image_geometries", [None for _ in range(batch_size)])
            names = batch["names"] if "names" in batch else [
                f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)
            ]

            pad_id = tokenizer.pad_token_id
            eos_id = tokenizer.eos_token_id

            for pred, gt, seq, name, geometry in zip(predictions, references, sequences, names, geometries):
                seq_list = seq.detach().cpu().tolist()

                if eos_id is not None and eos_id in seq_list:
                    gen_len = seq_list.index(eos_id) + 1
                elif pad_id is not None and pad_id in seq_list:
                    gen_len = seq_list.index(pad_id)
                else:
                    gen_len = len(seq_list)

                generated_lengths.append(int(gen_len))

                # Very fast structural proxy.
                if self.config.compute_valid_struct_seq:
                    try:
                        struct_seq = self.fast_metrics.score_structure_sequence(
                            gt_raw=gt,
                            pred_raw=pred,
                            key=name,
                        )
                        struct_seq_scores.append(float(struct_seq))
                    except Exception as exc:
                        print(f"Structure-sequence metric failed for {name}: {exc}")

                # Tree-based S-TEDS. This is the best metric for checkpoint selection
                # if full TEDS is too slow.
                if self.config.compute_valid_steds:
                    try:
                        steds = self.fast_metrics.score_steds(
                            gt_raw=gt,
                            pred_raw=pred,
                            key=name,
                        )
                        steds_scores.append(float(steds))
                    except Exception as exc:
                        print(f"S-TEDS failed for {name}: {exc}")

                # Full TEDS. Keep disabled during frequent validation unless needed.
                if self.config.compute_valid_teds:
                    try:
                        teds = self.fast_metrics.score_teds(
                            gt_raw=gt,
                            pred_raw=pred,
                            key=name,
                        )
                        teds_scores.append(float(teds))
                    except Exception as exc:
                        print(f"TEDS failed for {name}: {exc}")

                if self.config.compute_valid_box_metrics:
                    try:
                        scale_x = float(geometry.get("scale_x", 1.0)) if isinstance(geometry, dict) else 1.0
                        scale_y = float(geometry.get("scale_y", scale_x)) if isinstance(geometry, dict) else scale_x
                        box_metrics = compute_box_metrics_from_strings(
                            reference=gt,
                            prediction=pred,
                            image_scale_x=scale_x,
                            image_scale_y=scale_y,
                        ).to_dict(prefix="")
                        box_metric_dicts.append(box_metrics)
                    except Exception as exc:
                        print(f"Box metric failed for {name}: {exc}")

            processed_batches += 1
            pbar.update(1)

            postfix: dict[str, Any] = {
                "batches": processed_batches,
                "samples": len(generated_lengths),
            }

            if struct_seq_scores:
                postfix["struct_seq"] = f"{float(np.mean(struct_seq_scores)):.4f}"

            if steds_scores:
                postfix["steds"] = f"{float(np.mean(steds_scores)):.4f}"

            if teds_scores:
                postfix["teds"] = f"{float(np.mean(teds_scores)):.4f}"

            if box_metric_dicts:
                box_running = average_metric_dicts(box_metric_dicts)
                if "box_iou" in box_running:
                    postfix["box_iou"] = f"{box_running['box_iou']:.4f}"
                if "box_f1_50" in box_running:
                    postfix["box_f1_50"] = f"{box_running['box_f1_50']:.4f}"

            if generated_lengths:
                postfix["len"] = f"{float(np.mean(generated_lengths)):.1f}"

            pbar.set_postfix(**postfix)

        pbar.close()

        metrics: dict[str, float] = {}

        if struct_seq_scores:
            metrics["valid_struct_seq"] = float(np.mean(struct_seq_scores))

        if steds_scores:
            metrics["valid_steds"] = float(np.mean(steds_scores))

        if teds_scores:
            metrics["valid_teds"] = float(np.mean(teds_scores))

        if box_metric_dicts:
            for key, value in average_metric_dicts(box_metric_dicts).items():
                metrics[f"valid_{key}"] = float(value)

        if generated_lengths:
            metrics["valid_gen_len"] = float(np.mean(generated_lengths))

        metrics["valid_generation_batches"] = float(processed_batches)
        metrics["valid_generation_samples"] = float(len(generated_lengths))

        return metrics


    @torch.no_grad()
    def show_prediction_samples(self, loader=None, num_samples: int = 3, max_length: Optional[int] = None) -> None:
        if loader is None:
            loader = self.valid_loader
        if loader is None:
            print("No validation loader available for prediction samples.")
            return

        self.model.eval()
        tokenizer = self.model.decoder.tokenizer
        max_length = max_length or self.config.max_length
        sample_path = self.sample_dir / f"step_{self.global_step}.txt"

        printed = 0
        chunks: list[str] = []
        for batch in loader:
            images = batch["imgs"].to(self.device, non_blocking=True)
            image_attention_mask = batch.get("image_attention_mask")
            if image_attention_mask is not None:
                image_attention_mask = image_attention_mask.to(self.device, non_blocking=True)
            batch_size = images.size(0)
            input_ids, _ = self.model.build_input_ids_from_prompt("<html>", batch_size=batch_size, device=self.device)

            output = self.model.generate(
                images=images,
                input_ids=input_ids,
                image_attention_mask=image_attention_mask,
                max_length=max_length,
                use_cache=True,
                use_structure_bias=False,
                return_dict_in_generate=True,
            )
            sequences = extract_sequences(output)

            for i in range(batch_size):
                pred = tokenizer.decode(sequences[i], skip_special_tokens=False)
                gt = batch["raw_labels"][i]
                name = batch["names"][i] if "names" in batch else f"sample_{printed}"
                pred_clean = clean_tableseq_html(pred, remove_formatting_tags=True)
                gt_clean = clean_tableseq_html(gt, remove_formatting_tags=True)

                lines = ["=" * 120, f"step={self.global_step} sample={name}"]
                try:
                    lines.append(
                        f"S-TEDS={self.fast_metrics.score_steds(gt_raw=gt, pred_raw=pred, key=name):.4f}"
                    )
                except Exception as exc:
                    lines.append(f"S-TEDS failed: {exc}")

                if self.config.compute_valid_box_metrics:
                    try:
                        geometry = batch.get("image_geometries", [None for _ in range(batch_size)])[i]
                        scale_x = float(geometry.get("scale_x", 1.0)) if isinstance(geometry, dict) else 1.0
                        scale_y = float(geometry.get("scale_y", scale_x)) if isinstance(geometry, dict) else scale_x
                        box_metrics = compute_box_metrics_from_strings(
                            reference=gt,
                            prediction=pred,
                            image_scale_x=scale_x,
                            image_scale_y=scale_y,
                        ).to_dict(prefix="")
                        if "box_iou" in box_metrics:
                            lines.append(
                                "Boxes: "
                                f"IoU={box_metrics['box_iou']:.4f}, "
                                f"F1@0.5={box_metrics.get('box_f1_50', 0.0):.4f}, "
                                f"n_pred={int(box_metrics.get('n_pred_boxes', 0))}, "
                                f"n_gt={int(box_metrics.get('n_gt_boxes', 0))}"
                            )
                        else:
                            lines.append("Boxes: no coordinate boxes found in prediction/reference.")
                    except Exception as exc:
                        lines.append(f"Box metric failed: {exc}")

                if self.config.compute_valid_teds:
                    try:
                        lines.append(
                            f"TEDS={self.fast_metrics.score_teds(gt_raw=gt, pred_raw=pred, key=name):.4f}"
                        )
                    except Exception as exc:
                        lines.append(f"TEDS failed: {exc}")

                if self.config.show_coordinates_in_samples:
                    lines += ["\nPRED RAW WITH COORDS:", pred[:2000], "\nGT RAW WITH COORDS:", gt[:2000]]

                lines += ["\nPRED CLEAN:", pred_clean[:1500], "\nGT CLEAN:", gt_clean[:1500]]
                text = "\n".join(lines)
                print("\n" + text)
                chunks.append(text)

                printed += 1
                if printed >= num_samples:
                    with open(sample_path, "w", encoding="utf-8") as f:
                        f.write("\n\n".join(chunks))
                    print(f"Saved prediction samples: {sample_path}")
                    return

    def run_validation(self, phase: str = "valid") -> dict[str, float]:
        print(
            f"[{phase}] running validation at step {self.global_step} "
            f"with max_generation_batches={self.config.max_valid_batches_for_generation}, "
            f"compute_valid_loss={self.config.compute_valid_loss}, "
            f"compute_valid_struct_seq={self.config.compute_valid_struct_seq}, "
            f"compute_valid_steds={self.config.compute_valid_steds}, "
            f"compute_valid_teds={self.config.compute_valid_teds}, "
            f"compute_valid_box_metrics={self.config.compute_valid_box_metrics}."
        )
        metrics = self.validate_loss()
        metrics.update(self.validate_generation())
        if metrics:
            self.log_metrics(metrics, phase=phase)
        else:
            print(f"[{phase}] no validation metric was computed.")
        return metrics

    def maybe_save_best(self, metrics: dict[str, float]) -> None:
        metric_name = self.config.metric_for_best
        if metric_name not in metrics:
            print(
                f"[valid] metric_for_best='{metric_name}' was not computed. "
                "No best checkpoint update."
            )
            return

        value = float(metrics[metric_name])
        if self.best_metric is None or value > self.best_metric:
            previous = self.best_metric
            self.best_metric = value
            print(f"[valid] new best {metric_name}: {value:.4f} (previous={previous})")
            self.save_training_checkpoint(name="best")
        else:
            print(f"[valid] {metric_name}={value:.4f}; best remains {self.best_metric:.4f}.")

    def save_training_checkpoint(self, name: str = "last", epoch: Optional[int] = None) -> None:
        """
        Save a full resumable training checkpoint.

        Only two checkpoint files are used during training:
            checkpoints/last.pt
            checkpoints/best.pt

        No step_<N>.pt files are created, to avoid filling the disk.
        """
        path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "epoch": int(self.current_epoch if epoch is None else epoch),
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_config": self.checkpoint_training_config,
            "data_config": self.checkpoint_data_config,
            "best_metric": self.best_metric,
            "best_metric_name": self.config.metric_for_best,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved training checkpoint: {path}")


    def train(self) -> None:
        if self.config.validate_at_start:
            metrics = self.run_validation(phase="valid_start")
            self.maybe_save_best(metrics)
            self.show_prediction_samples(num_samples=self.config.num_prediction_samples)

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            running: dict[str, list[float]] = {
                "loss": [],
                "ce_loss": [],
                "token_acc": [],
                "grad_norm": [],
                "tokens_per_sec": [],
                "noise_changed": [],
                "noise_eligible": [],
                "coord_token_acc": [],
                "x_token_acc": [],
                "y_token_acc": [],
                "coord_token_count": [],
            }
            pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch}")

            for batch in pbar:
                stats = self.train_step(batch)
                for key in running:
                    if key in stats:
                        running[key].append(float(stats[key]))

                noise_rate = 0.0
                if stats["noise_eligible"] > 0:
                    noise_rate = stats["noise_changed"] / stats["noise_eligible"]

                postfix = {
                    "step": self.global_step,
                    "loss": f"{stats['loss']:.4f}",
                    "acc": f"{stats['token_acc']:.4f}",
                    "lr": f"{stats['lr']:.2e}",
                    "noise": f"{noise_rate:.3f}",
                }
                if "coord_token_acc" in stats:
                    postfix["coord_acc"] = f"{stats['coord_token_acc']:.4f}"
                pbar.set_postfix(**postfix)

                if self.global_step % self.config.log_every_n_steps == 0:
                    now = time.time()
                    step_delta = max(self.global_step - self.last_log_step, 1)
                    time_delta = max(now - self.last_log_time, 1e-6)
                    log_metrics = {
                        "epoch": float(epoch),
                        "loss": float(np.mean(running["loss"][-self.config.log_every_n_steps:])),
                        "ce_loss": float(np.mean(running["ce_loss"][-self.config.log_every_n_steps:])),
                        "token_acc": float(np.mean(running["token_acc"][-self.config.log_every_n_steps:])),
                        "grad_norm": float(np.mean(running["grad_norm"][-self.config.log_every_n_steps:])),
                        "lr": float(stats["lr"]),
                        "steps_per_sec": float(step_delta / time_delta),
                        "tokens_per_sec": float(np.mean(running["tokens_per_sec"][-self.config.log_every_n_steps:])),
                        "noise_changed": float(np.sum(running["noise_changed"][-self.config.log_every_n_steps:])),
                        "noise_eligible": float(np.sum(running["noise_eligible"][-self.config.log_every_n_steps:])),
                    }
                    for metric_name in ("coord_token_acc", "x_token_acc", "y_token_acc"):
                        if running.get(metric_name):
                            log_metrics[metric_name] = float(
                                np.mean(running[metric_name][-self.config.log_every_n_steps:])
                            )
                    if running.get("coord_token_count"):
                        log_metrics["coord_token_count"] = float(
                            np.sum(running["coord_token_count"][-self.config.log_every_n_steps:])
                        )

                    self.log_metrics(log_metrics, phase="train")
                    self.last_log_time = now
                    self.last_log_step = self.global_step

                if (
                    self.config.generate_samples_every_n_steps
                    and self.global_step % self.config.generate_samples_every_n_steps == 0
                ):
                    self.show_prediction_samples(num_samples=self.config.num_prediction_samples)

                if self.config.eval_every_n_steps and self.global_step % self.config.eval_every_n_steps == 0:
                    metrics = self.run_validation(phase="valid")
                    self.maybe_save_best(metrics)

                if self.config.save_every_n_steps and self.global_step % self.config.save_every_n_steps == 0:
                    self.save_training_checkpoint(name="last")

                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break

            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break

        if self.config.validate_at_end:
            metrics = self.run_validation(phase="valid_end")
            self.maybe_save_best(metrics)
        if self.config.save_at_end:
            self.save_training_checkpoint(name="last")
